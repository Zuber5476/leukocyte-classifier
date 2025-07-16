# 🧬 Leukocyte Classifier model

A deep learning–powered web app that classifies white blood cells (WBCs) into four major types: **Neutrophil**, **Eosinophil**, **Monocyte**, and **Lymphocyte**.  
Built with TensorFlow

---

## 🔍 Features

- 🧠 Trained CNN model on real blood cell images
- 📊 Predicts WBC class with confidence scores
- ⚙️ Lightweight TensorFlow Lite model (.tflite)
- 🌐 Easy-to-use web interface (Streamlit)
- 🧾 Optionally supports Grad-CAM for explainability

---

## 📁 Project Structure

```
leukocyte_classifier/
├── model/
│   └── leukocyte_model.tflite      # Trained and converted model
├── streamlit_app/
│   └── app.py                      # Streamlit UI
├── requirements.txt                # Required Python packages
├── README.md                       # This file
```

---

## 🚀 How to Run Locally

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the app**
```bash
streamlit run streamlit_app/app.py
```

3. **Visit**  
[http://localhost:8501](http://localhost:8501)

---

## 🌍 How to Deploy Online (Streamlit Cloud)

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **“New app”** and:
   - Choose this repo
   - Set `streamlit_app/app.py` as the main file
3. ✅ Your app will be publicly available

---

## 📦 Requirements

```txt
streamlit
tensorflow
pillow
numpy
```

---

## 🧪 Sample Image

You can upload any WBC microscope image like the one below:


![Uploading images.jpg…]()

---

## 🧠 Model Details

- Input size: `128x128`
- Output: Softmax over 4 classes
- Accuracy: ~85–90% (based on Kaggle WBC dataset)
- Trained using: TensorFlow/Keras on Google Colab

---

## 👨‍💻 Author

- Name:
- Email:
- GitHub: 
---

## 📜 License

MIT License – free to use for research and education.
