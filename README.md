**Sign Language Detection Project**

This project aims to detect and recognize sign language gestures using computer vision and machine learning techniques. It utilizes the MediaPipe library for hand detection and tracking, as well as a trained machine learning model to recognize gestures.

**Requirements**
Python 3.x
OpenCV
MediaPipe
TensorFlow/Keras
NumPy
pickle
gTTS (Google Text-to-Speech) - for macOS users

**Installation**
1. Clone the repository to your local machine:

git clone https://github.com/nrrpatel/sign-language-detection.git

2. Navigate to the project directory:

cd sign-language-detection

3. Install the required Python dependencies:

pip install -r requirements.txt

4. Usage:

Ensure your webcam is connected and properly configured.
Run the main script inference_classifier.py:


When a gesture is detected, the corresponding character or word will be displayed on the screen and announced (if supported by your system).

**Supported Gestures**
The model currently supports recognition of the following gestures:

A-Z letters

**Model Training**
The machine learning model used for gesture recognition was trained on a dataset of hand gesture images. The training process involved preprocessing the images, extracting relevant features using MediaPipe, and training a convolutional neural network (CNN) using TensorFlow/Keras.

Credits
This project was developed by Nikunj Patel and should not be recreated without the permission of the user. 
