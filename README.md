# Facial Emotion Recognition Using CNN

A desktop app that reads your face from a webcam and shows how you feel as an emoji. It uses a CNN trained on the FER2013 dataset to detect 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad and Surprised.

## How it works

1. OpenCV opens the webcam and finds faces with a Haar cascade.
2. Each face is cropped, turned to grayscale and resized to 48x48.
3. The CNN predicts the emotion.
4. A Tkinter window shows the live video, the emotion name and a matching emoji.

## Model

| Layer | Details |
| --- | --- |
| Conv2D + MaxPooling | 32 filters, 3x3 |
| Conv2D + MaxPooling | 64 filters, 3x3 |
| Dense | 128 units, ReLU |
| Output | 7 units, Softmax |

* Dataset: FER2013 (28,709 training images, 7,178 test images)
* Input: 48x48 grayscale
* Optimizer: Adam, categorical cross entropy loss, batch size 64

## Project structure

```
frontend_app.py                  Webcam app with Tkinter
main_train.py                    Downloads FER2013, trains the CNN and saves the model
emotion_recognition_model.keras  Trained model
setup_assets.py                  Downloads the emoji images
emojis/                          Emoji images for each emotion
create_dummy_data.py             Makes fake images for quick testing
create_dummy_model.py            Makes an untrained model for quick testing
docs/                            Project report
```

## How to run

1. Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Start the app:
   ```
   python frontend_app.py
   ```

The trained model is already in the repo. To train it again, run `python main_train.py`. It downloads FER2013 and builds the `data/` folder for you.

Do not run `create_dummy_model.py` unless you only want to test the app. It replaces the trained model with an untrained one.

## Tools

Python, TensorFlow, Keras, OpenCV, Tkinter, Pillow, NumPy.
