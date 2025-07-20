# 📧 Email Spam Detector

A machine learning-powered web application that intelligently classifies email or SMS messages as **Spam** or **Not Spam**, built using **Python**, **scikit-learn**, and **Streamlit**.

---

## 🚀 Overview

This project uses a Logistic Regression model trained on the widely-used [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) to identify unsolicited or harmful messages. The model processes raw text using TF-IDF vectorization and delivers predictions through a clean, user-friendly Streamlit interface.

- ⚙️ **Model:** Logistic Regression  
- 🧠 **Feature Extraction:** TF-IDF Vectorizer  
- 📊 **Accuracy:** ~94% on test data  
- 🌐 **Interface:** Streamlit App  

---

## 🖥️ Demo

Paste a message like:

> *"Congratulations! You have won a $1,000 gift card. Click here to claim now!"*

and the app will classify it as **Spam**.

Try another:

> *"Hey, just checking in about tomorrow's meeting schedule."*

and it will return **Not Spam**.

---

## 📁 Project Structure

email-spam-detector/
├── app/
│ └── app.py # Streamlit application
├── data/
│ └── spam.csv # Raw dataset
├── notebooks/
│ ├── spam_preprocessing.ipynb # Model training notebook
│ ├── spam_classifier.pkl # Trained model
│ └── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sayak-Ghosh99/email-spam-detector.git
   cd email-spam-detector
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app/app.py
📈 Model Performance
Metric	Value
Accuracy	94.4%
Precision	97% (spam)
Recall	61% (spam)
F1-Score	75% (spam)

🛠️ Technologies Used
Python 3.x

Scikit-learn

Pandas & NumPy

Streamlit

Joblib

👤 Author
Sayak Ghosh
🔗 GitHub: @Sayak-Ghosh99
📫 Email: ghoshsayak2017@gmail.com
