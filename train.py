from data_preprocessing import train_generator, val_generator
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = create_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)
