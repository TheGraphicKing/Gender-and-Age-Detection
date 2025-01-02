import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing function
def preprocess_image(image_path, img_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    return image

# Data augmentation generator
def get_data_generator():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
```

---

#### `utils/config.py`
```python
# Configuration file

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
```
