"""
MindSpore-based Model for Skin Lesion Classification
Improved architecture with residual connections and attention mechanisms
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import HeNormal, Normal
import config


class ChannelAttention(nn.Cell):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = ops.AdaptiveAvgPool2D(1)
        self.max_pool = ops.AdaptiveMaxPool2D(1)
        
        self.fc = nn.SequentialCell([
            nn.Dense(channels, channels // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channels // reduction, channels, has_bias=False)
        ])
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out)
        return out.view(b, c, 1, 1) * x


class SpatialAttention(nn.Cell):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size//2, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = ops.Concat(axis=1)
        
    def construct(self, x):
        avg_out = ops.ReduceMean(keep_dims=True)(x, 1)
        max_out = ops.ReduceMax(keep_dims=True)(x, 1)
        out = self.concat([avg_out, max_out])
        out = self.sigmoid(self.conv(out))
        return out * x


class CBAM(nn.Cell):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def construct(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Cell):
    """Residual Block with Batch Normalization and Dropout"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, has_bias=False,
                               weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, has_bias=False,
                               weight_init=HeNormal())
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(keep_prob=1.0-dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm2d(out_channels)
            ])
        
        self.cbam = CBAM(out_channels)
        
    def construct(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class SkinLesionClassifier(nn.Cell):
    """
    Improved Skin Lesion Classification Model with:
    - Residual connections
    - Attention mechanisms (CBAM)
    - Batch normalization
    - Dropout for regularization
    """
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=0.5):
        super(SkinLesionClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling
        self.avgpool = ops.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        
        # Fully connected layers with dropout
        self.fc1 = nn.Dense(512, 256, weight_init=HeNormal())
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(keep_prob=1.0-dropout_rate)
        
        self.fc2 = nn.Dense(256, 128, weight_init=HeNormal())
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(keep_prob=1.0-dropout_rate)
        
        # Output layer
        self.fc_out = nn.Dense(128, num_classes, weight_init=Normal(0.01))
        self.softmax = nn.Softmax(axis=1)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and flatten
        x = self.avgpool(x)
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc_out(x)
        
        return x


class EfficientNetB0MindSpore(nn.Cell):
    """
    Lightweight model based on EfficientNet principles
    Better for faster training and inference
    """
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=0.4):
        super(EfficientNetB0MindSpore, self).__init__()
        
        # Stem
        self.stem = nn.SequentialCell([
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, 
                     has_bias=False, weight_init=HeNormal()),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ])
        
        # MBConv blocks (simplified version)
        self.blocks = nn.SequentialCell([
            self._make_mbconv_block(32, 16, 1, 1),
            self._make_mbconv_block(16, 24, 2, 6),
            self._make_mbconv_block(24, 40, 2, 6),
            self._make_mbconv_block(40, 80, 3, 6),
            self._make_mbconv_block(80, 112, 3, 6),
            self._make_mbconv_block(112, 192, 4, 6),
            self._make_mbconv_block(192, 320, 1, 6)
        ])
        
        # Head
        self.head = nn.SequentialCell([
            nn.Conv2d(320, 1280, kernel_size=1, has_bias=False, weight_init=HeNormal()),
            nn.BatchNorm2d(1280),
            nn.ReLU()
        ])
        
        self.avgpool = ops.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(keep_prob=1.0-dropout_rate)
        self.fc = nn.Dense(1280, num_classes, weight_init=Normal(0.01))
        
    def _make_mbconv_block(self, in_channels, out_channels, num_repeat, expand_ratio):
        """Create MBConv block"""
        layers = []
        for i in range(num_repeat):
            stride = 2 if i == 0 and in_channels != out_channels else 1
            layers.append(
                MBConvBlock(in_channels if i == 0 else out_channels, 
                           out_channels, stride, expand_ratio)
            )
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MBConvBlock(nn.Cell):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        if expand_ratio != 1:
            # Expand
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                     group=hidden_dim, has_bias=False, weight_init=HeNormal()),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, has_bias=False, weight_init=HeNormal()),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.SequentialCell(layers)
        
    def construct(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


def create_model(model_type='resnet', num_classes=config.NUM_CLASSES):
    """
    Factory function to create different model architectures
    
    Args:
        model_type: 'resnet' or 'efficientnet'
        num_classes: Number of output classes
        
    Returns:
        MindSpore model
    """
    if model_type == 'resnet':
        return SkinLesionClassifier(num_classes=num_classes)
    elif model_type == 'efficientnet':
        return EfficientNetB0MindSpore(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Set context
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, 
                         device_target=config.MINDSPORE_DEVICE_TARGET)
    
    # Create model
    model = create_model(model_type='resnet')
    
    # Test with random input
    dummy_input = Tensor(mindspore.numpy.randn(1, 3, 224, 224), mindspore.float32)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum([p.size for p in model.trainable_params()])}")
