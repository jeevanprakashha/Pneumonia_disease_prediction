from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/Users/jeevanprakash/Documents/deep_project/chest_xray_model.h5')

@app.route('/', methods=['GET'])
def index():
    # Render the main page
    return render_template('index.html')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap_on_image(heatmap, original_image_path, intensity=0.5, colormap=cv2.COLORMAP_VIRIDIS):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, colormap)

    superimposed_image = heatmap_resized * intensity + original_image
    superimposed_image = np.clip(superimposed_image, 0, 255).astype(np.uint8)
    return superimposed_image

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']

    basepath = os.path.dirname(__file__)
    static_folder = os.path.join(basepath, 'static', 'uploads')

    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    file_path = os.path.join(static_folder, secure_filename(f.filename))
    f.save(file_path)

    img = load_img(file_path, target_size=(180, 180))
    img = img_to_array(img)
    img_prep = np.expand_dims(img, axis=0)
    img_prep = img_prep / 255.0

    prediction = model.predict(img_prep)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

    heatmap = make_gradcam_heatmap(img_prep, model, last_conv_layer_name='conv2d_2')
    superimposed_img = superimpose_heatmap_on_image(heatmap, file_path)

    heatmap_path = os.path.join(static_folder, 'heatmap_' + secure_filename(f.filename))
    cv2.imwrite(heatmap_path, superimposed_img)

    heatmap_url = url_for('static', filename='uploads/heatmap_' + secure_filename(f.filename))
    return render_template('result.html', result=result, heatmap_url=heatmap_url)

if __name__ == '__main__':
    app.run(debug=True)
