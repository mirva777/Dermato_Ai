"""
Setup script to verify installation and prepare environment
"""

import os
import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.9+")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'flask': 'flask',
        'requests': 'requests'
    }
    
    optional = {
        'tensorflow': 'tensorflow',
        'mindspore': 'mindspore',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    all_ok = True
    
    # Check required
    print("\nRequired packages:")
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")
            all_ok = False
    
    # Check optional
    print("\nOptional packages:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} - Not installed (optional)")
    
    return all_ok


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    dirs = [
        'uploads',
        'logs',
        'checkpoints',
        'HAM10000_images'
    ]
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created {dir_name}/")
        else:
            print(f"✓ {dir_name}/ already exists")


def check_dataset():
    """Check if dataset is available"""
    print("\nChecking dataset...")
    
    metadata_exists = os.path.exists('HAM10000_metadata.csv')
    images_exist = os.path.exists('HAM10000_images') and \
                   len(os.listdir('HAM10000_images')) > 0
    
    if metadata_exists:
        print("✓ HAM10000_metadata.csv found")
    else:
        print("✗ HAM10000_metadata.csv not found")
        print("  Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    
    if images_exist:
        num_images = len([f for f in os.listdir('HAM10000_images') 
                         if f.endswith(('.jpg', '.png'))])
        print(f"✓ Found {num_images} images in HAM10000_images/")
    else:
        print("✗ HAM10000_images/ is empty or doesn't exist")
        print("  Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    
    return metadata_exists and images_exist


def check_models():
    """Check if trained models exist"""
    print("\nChecking models...")
    
    tf_model = os.path.exists('model.h5')
    ms_model = os.path.exists('model_mindspore.ckpt')
    
    if tf_model:
        print("✓ TensorFlow model (model.h5) found")
    else:
        print("⚠ TensorFlow model not found - Need to train")
    
    if ms_model:
        print("✓ MindSpore model (model_mindspore.ckpt) found")
    else:
        print("⚠ MindSpore model not found - Need to train")
    
    return tf_model or ms_model


def install_mindspore():
    """Provide instructions for MindSpore installation"""
    print("\n" + "="*80)
    print("MindSpore Installation Guide")
    print("="*80)
    print("\nMindSpore installation depends on your system:")
    print("\n1. CPU (Windows/Linux):")
    print("   pip install mindspore==2.2.0")
    
    print("\n2. GPU (CUDA 11.6):")
    print("   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/gpu/x86_64/cuda-11.6/mindspore_gpu-2.2.0-cp39-cp39-linux_x86_64.whl")
    
    print("\n3. Ascend (Huawei NPU):")
    print("   See: https://www.mindspore.cn/install/en")
    
    print("\nFor more details: https://www.mindspore.cn/install/en")
    print("="*80)


def main():
    """Main setup function"""
    print("="*80)
    print("Dermato AI - Setup and Verification")
    print("="*80)
    
    # Check Python
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Some required packages are missing")
        print("\nInstall with:")
        print("  pip install -r requirements_updated.txt")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create directories
    create_directories()
    
    # Check dataset
    dataset_ok = check_dataset()
    if not dataset_ok:
        print("\n⚠ Dataset not found - Download required before training")
    
    # Check models
    models_ok = check_models()
    if not models_ok:
        print("\n⚠ No trained models found")
        print("  Train with: python train_mindspore.py --model resnet --epochs 30")
    
    # MindSpore installation guide
    try:
        import mindspore
        print(f"\n✓ MindSpore {mindspore.__version__} is installed")
    except ImportError:
        print("\n⚠ MindSpore not installed")
        response = input("\nShow MindSpore installation guide? (y/n): ")
        if response.lower() == 'y':
            install_mindspore()
    
    # Summary
    print("\n" + "="*80)
    print("Setup Summary")
    print("="*80)
    
    if dataset_ok and models_ok:
        print("✓ Ready to run!")
        print("\nQuick start:")
        print("  1. Start API: python main_improved.py")
        print("  2. Test inference: python inference.py --image <path>")
    elif dataset_ok:
        print("⚠ Dataset ready - Train models to start")
        print("\nNext steps:")
        print("  1. Train: python train_mindspore.py --model resnet")
        print("  2. Evaluate: python evaluate_improved.py --framework mindspore")
    else:
        print("⚠ Download dataset to continue")
        print("\nNext steps:")
        print("  1. Download HAM10000 dataset")
        print("  2. Run setup again: python setup.py")
    
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
