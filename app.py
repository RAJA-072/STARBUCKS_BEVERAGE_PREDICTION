from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]

        return render_template('index.html', prediction=pred_class, img_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
