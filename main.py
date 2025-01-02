import os
from models.model import build_model
from utils.preprocess import preprocess_image, get_data_generator
from utils.config import IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import glob

# Load dataset
images = []
labels = []

for label, folder in enumerate(os.listdir('data/dataset')):
    for image_path in glob.glob(f"data/dataset/{folder}/*.jpg"):
        images.append(preprocess_image(image_path, IMG_SIZE))
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

data_gen = get_data_generator()
data_gen.fit(X_train)

# Build model
model = build_model(input_shape=IMG_SIZE + (3,), num_classes=len(set(labels)))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE
)

# Save model
model.save("gender_age_model.h5")
```

---

