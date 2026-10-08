import os
import json
import os
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import urllib.request
import csv
import numpy as np
from PIL import Image

# --- DOWNLOAD AND PREPARE DATA ---
if not os.path.exists('data'):
    csv_url = 'https://raw.githubusercontent.com/mohamedamine99/FER-2013-Dataset/master/fer2013.csv'
    csv_file = 'fer2013.csv'
    urllib.request.urlretrieve(csv_url, csv_file)

    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    for emotion in range(7):
        os.makedirs(f'data/train/{emotion}', exist_ok=True)
        os.makedirs(f'data/test/{emotion}', exist_ok=True)

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            emotion = int(row[0])
            pixels = np.array([int(p) for p in row[1].split()], dtype=np.uint8).reshape(48, 48)
            usage = row[2]
            img = Image.fromarray(pixels, mode='L')
            folder = 'train' if usage == 'Training' else 'test'
            # Save with unique name
            import random
            filename = f'{random.randint(0,100000)}.png'
            img.save(f'data/{folder}/{emotion}/{filename}')

    print("Data prepared.")

train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_datagenerator.flow_from_directory(
    'data/train', target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')

test_generator = test_datagenerator.flow_from_directory(
    'data/test', target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')

# --- DATA GENERATORS (Added Data Augmentation) ---
train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_datagenerator.flow_from_directory(
    'data/train', target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')

test_generator = test_datagenerator.flow_from_directory(
    'data/test', target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')

# --- IMPROVED CNN ARCHITECTURE ---
model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten & Dense Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.001), 
              metrics=['accuracy'])

# --- TRAIN AND SAVE ---
# Increasing epochs to 1 for quick training on real data
model.fit(train_generator, epochs=1, validation_data=test_generator)
model.save('emotion_recognition_model.keras')