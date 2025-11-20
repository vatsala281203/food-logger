import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

DATA_DIR = r"C:\Users\HP\Desktop\foodtask\food-logger\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224,224)
BATCH = 32
EPOCHS = 10
MODEL_OUT = r"C:\Users\HP\Desktop\foodtask\food-logger\models\food_model.h5"
LABELS_OUT = r"C:\Users\HP\Desktop\foodtask\food-logger\models\labels.json"

train_gen = ImageDataGenerator(rescale=1./255,rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH)
val_flow   = val_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
out = layers.Dense(train_flow.num_classes, activation='softmax')(x)

model = models.Model(base.input, out)
model.compile(optimizer=optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=4, restore_best_weights=True, monitor='val_loss')
]

history = model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS, callbacks=callbacks)

labels = {str(v): k for k, v in train_flow.class_indices.items()}
with open(LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

