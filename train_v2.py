import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_P1  = 10
EPOCHS_P2  = 15

print("\n[1/6] Loading PlantVillage dataset...")
(train_raw, val_raw, test_raw), info = tfds.load(
    'plant_village',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
    shuffle_files=True
)
class_names = info.features['label'].names
NUM_CLASSES = len(class_names)
print(f"✅ Dataset loaded! Total classes: {NUM_CLASSES}")

print("\n[2/6] Setting up preprocessing...")
def preprocess_train(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=30)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = preprocess_input(image)
    return image, label

def preprocess_eval(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_raw.map(preprocess_train, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (val_raw.map(preprocess_eval, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))
test_ds  = (test_raw.map(preprocess_eval, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))
print("✅ Preprocessing ready!")

print("\n[3/6] Building MobileNetV2 model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("✅ Model built!")

print("\n[4/6] Phase 1: Training classification head...")
cb_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('best_phase1.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1, callbacks=cb_p1)
print(f"✅ Phase 1 done! Best Val Accuracy: {max(history1.history['val_accuracy'])*100:.2f}%")

print("\n[5/6] Phase 2: Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cb_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('plant_disease_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]
history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P2, callbacks=cb_p2)
print(f"✅ Phase 2 done! Best Val Accuracy: {max(history2.history['val_accuracy'])*100:.2f}%")

print("\n[6/6] Final evaluation...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n🎯 FINAL TEST ACCURACY: {test_acc*100:.2f}%")

with open('class_names.json', 'w') as f:
    json.dump(list(class_names), f, indent=2)

model.save('plant_disease_model.keras')

print("\n✅ DONE! Download these 2 files from left sidebar:")
print("   1. plant_disease_model.keras")
print("   2. class_names.json")