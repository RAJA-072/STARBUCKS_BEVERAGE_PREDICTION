## ☕ Starbucks Beverage Image Classifier

A visually stunning web application that predicts the type of Starbucks beverage based on an uploaded image. Powered by a custom-trained CNN model using TensorFlow and Flask.

![image](https://github.com/user-attachments/assets/269e4b37-0395-403d-8d59-8d2b0b7b58ec)

---

### 🧠 Labels

The model is trained to classify Starbucks drinks into four categories:

- 🟠 `Citrus`
- 🌈 `Unicorn`
- 🤎 `Latte`
- 🍫 `Mocha`

---

### 🚀 Features

- 📷 Upload your beverage image and get an instant prediction
- 🖼️ Clean, glassmorphic HTML/CSS UI
- ⚡ Fast and accurate model built with TensorFlow/Keras
- 🌐 Flask-based lightweight backend
- 💡 Easily extensible to more categories or data

---

### 🗂️ Dataset Structure

```
train/
├── citrus/
│   ├── 1.jpg
│   ├── ...
├── unicorn/
├── latte/
├── mocha/
```

> Each folder contains 10 `.jpeg` images of the respective beverage.

---

### ⚙️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- Flask
- HTML & CSS (Custom UI)
- PyCharm / Colab Compatible

---

### 🔧 Getting Started

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/starbucks-beverage-classifier.git
cd starbucks-beverage-classifier
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Train the Model (Optional)

```bash
python train_model.py
```

> Skippable if `model.h5` and `class_indices.json` already exist.

#### 4. Run the Web App

```bash
python app.py
```

#### 5. Open in Your Browser

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 🖼️ Live Demo (Optional)

![Prediction Example](https://via.placeholder.com/600x300.png?text=Predicted:+Mocha)  
> Replace with an actual screenshot of prediction output.

---

### 📁 Project Structure

| File/Folder              | Description                            |
|--------------------------|----------------------------------------|
| `train/`                 | Training images organized by category  |
| `app.py`                 | Flask server for frontend/backend      |
| `train_model.py`         | Python script to train & save model    |
| `templates/index.html`   | Frontend UI with upload and display    |
| `model.h5`               | Trained Keras model                    |
| `class_indices.json`     | Mapping of classes to indices          |

---
