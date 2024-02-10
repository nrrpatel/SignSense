Sign Language Detection Project
This project aims to detect and recognize sign language gestures using computer vision and machine learning techniques. It utilizes the MediaPipe library for hand detection and tracking, as well as a trained machine learning model to recognize gestures.

**Requirements**
Python 3.x
OpenCV
MediaPipe
TensorFlow/Keras
NumPy
pickle
gTTS (Google Text-to-Speech) - for macOS users
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/sign-language-detection.git
Navigate to the project directory:

bash
Copy code
cd sign-language-detection
Install the required Python dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Ensure your webcam is connected and properly configured.

Run the main script inference_classifier.py:

bash
Copy code
python inference_classifier.py
The script will use your webcam feed to detect and recognize sign language gestures in real-time.

When a gesture is detected, the corresponding character or word will be displayed on the screen and announced (if supported by your system).

Supported Gestures
The model currently supports recognition of the following gestures:

A-Z letters
Model Training
The machine learning model used for gesture recognition was trained on a dataset of hand gesture images. The training process involved preprocessing the images, extracting relevant features using MediaPipe, and training a convolutional neural network (CNN) using TensorFlow/Keras.

Credits
This project was developed by Nikunj Patel
