from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import os

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the CIFAR-10 model
model = load_model("model.h5")

# CIFAR-10 class labels
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Webcam capture flag
camera = cv2.VideoCapture(0)  # 0 = default webcam

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to CIFAR-10 size
    img = np.array(img).astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Generate webcam frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Capture image from webcam
        ret, frame = camera.read()
        if ret:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
            cv2.imwrite(image_path, frame)

            # Preprocess image
            img = preprocess_image(image_path)
            prediction = model.predict(img)[0]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            return render_template('index.html', image=image_path, label=predicted_class, confidence=confidence)

    return render_template('index.html', image=None)

# Route to stream the webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True)
