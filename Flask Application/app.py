import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize Flask app
app = Flask(__name__)

# Define the VGG19-based model
base_model = VGG19(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)  # Ensure matching shape with weights file
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Load model weights
try:
    model_03.load_weights('vgg_unfrozen.h5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

print('Model loaded. Check http://127.0.0.1:5000/')

# Define helper functions
def get_class_name(class_no):
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"

def get_result(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Image not loaded. Please check the file path or format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    image = Image.fromarray(image)
    image = image.resize((128, 128))
    image = np.array(image) / 255.0  # Normalize the image
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    return np.argmax(result, axis=1)[0]

# Define routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction result
        value = get_result(file_path)
        result = get_class_name(value)
        return result
    return "Upload a file to predict."

if __name__ == '__main__':
    app.run(debug=True)
