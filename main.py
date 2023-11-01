from flask import Flask, render_template, request, send_file, url_for
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import io

app = Flask(__name__)
model = tf.keras.models.load_model('models/model_1000_epochs.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Reuse existing Flask request methods
        uploaded_img = request.files['image']
        img = Image.open(uploaded_img)
        img = img.resize((256, 256))
        img = np.array(img)/255.
        high_res_img = model.predict(np.expand_dims(img, axis=0))[0]
        high_res_img = Image.fromarray(np.uint8(np.clip(high_res_img*255, 0, 255))).convert('RGB')
        img_buf = io.BytesIO()
        high_res_img.save(img_buf, format='JPEG')
        img_buf.seek(0)
        return send_file(img_buf, mimetype='image/jpeg', as_attachment=True, download_name="output_file.jpg")

    return render_template('home.html')

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')


if __name__ == '__main__':
    app.run(debug=True)