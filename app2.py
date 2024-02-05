from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import nibabel as nib
from scipy import ndimage

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/Users/jeevanprakash/Documents/deep_project/3d_image_classification.h5')

# Define functions for processing 3D CT scans
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume

@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Process the CT scan
    processed_scan = process_scan(file_path)
    processed_scan_reshaped = np.expand_dims(processed_scan, axis=0)
    processed_scan_reshaped = np.expand_dims(processed_scan_reshaped, axis=-1)

    # Predict
    prediction = model.predict(processed_scan_reshaped)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

    return render_template('result1.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
