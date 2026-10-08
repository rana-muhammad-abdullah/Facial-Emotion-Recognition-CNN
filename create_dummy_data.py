import os
import numpy as np
from PIL import Image

# Create data directories
os.makedirs('data/train/Angry', exist_ok=True)
os.makedirs('data/train/Disgusted', exist_ok=True)
os.makedirs('data/train/Fearful', exist_ok=True)
os.makedirs('data/train/Happy', exist_ok=True)
os.makedirs('data/train/Neutral', exist_ok=True)
os.makedirs('data/train/Sad', exist_ok=True)
os.makedirs('data/train/Surprised', exist_ok=True)

os.makedirs('data/test/Angry', exist_ok=True)
os.makedirs('data/test/Disgusted', exist_ok=True)
os.makedirs('data/test/Fearful', exist_ok=True)
os.makedirs('data/test/Happy', exist_ok=True)
os.makedirs('data/test/Neutral', exist_ok=True)
os.makedirs('data/test/Sad', exist_ok=True)
os.makedirs('data/test/Surprised', exist_ok=True)

# Create dummy grayscale images (48x48) with different mean values for each emotion
means = {"Angry": 25, "Disgusted": 60, "Fearful": 95, "Happy": 130, "Neutral": 165, "Sad": 200, "Surprised": 235}

for emotion in ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']:
    mean = means[emotion]
    for i in range(200):  # more images
        dummy_image = np.random.normal(mean, 15, (48, 48)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(dummy_image)
        img.save(f'data/train/{emotion}/{i}.png')
    for i in range(100):  # for test
        dummy_image = np.random.normal(mean, 15, (48, 48)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(dummy_image)
        img.save(f'data/test/{emotion}/{i}.png')

print("Dummy data created with different means.")