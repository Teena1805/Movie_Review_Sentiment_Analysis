# 🎬 Movie Review Sentiment Analysis

A Deep Learning-based web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**. The system uses an **LSTM model** trained on 50,000 labeled reviews from the IMDB dataset, with a simple Flask-based frontend for real-time analysis.

---

## 📌 Features

- ✅ Accepts custom movie review text
- ✅ Predicts sentiment (Positive / Negative)
- ✅ Displays prediction confidence
- ✅ Preprocessed with Tokenizer & Padding
- ✅ Trained LSTM model using TensorFlow & Keras
- ✅ Accuracy ~87.1%, Precision ~86.7%, Recall ~87.8%, F1-score ~87.3%, ROC AUC ~93.8%
- ✅ Simple and interactive web interface using Flask

---

## 🚀 Live Demo

*(Optional — add link if hosted on Heroku, Render, or locally hosted demo GIF)*

---

## 🧠 Objective

The goal of this project is to automatically detect the **sentiment of movie reviews** using **Natural Language Processing (NLP)** and **Deep Learning**, making it possible to automate tasks like:

- Classifying user feedback
- Moderating review platforms
- Monitoring social media or comment sections

---

## 📂 Project Structure
```plaintext
project_root/
├── app.py # Flask backend to serve predictions
├── main.py # Script for training and evaluating the model
├── data/
│ └── IMDB_Dataset.csv # Dataset with 50,000 reviews
├── model/
│ ├── lstm_sentiment_model.h5 # Trained LSTM model
│ ├── tokenizer.joblib # Tokenizer for preprocessing
│ └── label_encoder.joblib # Label encoder
├── static/
│ └── images/
│ └── movies.png # Background image or emoji-related media
├── templates/
│ └── index.html # Frontend UI (rendered with Flask)
```

---

## 🎯 Objective

To create a system that automatically detects the **sentiment** (positive or negative) of movie reviews using **Natural Language Processing (NLP)** and **Deep Learning**, and provides a clean, interactive web interface for real-time analysis.

---

## 🛠️ Technologies Used

### 🔹 Frontend
- **HTML5** + **Jinja2 templates**
- **CSS** (via `style.css` if present)
- **Image assets** from `/static/images/` (`movies.png`)

### 🔹 Backend
- **Flask** for serving the web app
- **TensorFlow + Keras** for building the LSTM model
- **Scikit-learn** for evaluation metrics
- **Pandas / NumPy** for data handling
- **Joblib** for saving/loading tokenizer and encoders

---

## 📊 Model Performance

| Metric     | Value    |
|------------|----------|
| Accuracy   | 87.13%   |
| Precision  | 86.74%   |
| Recall     | 87.89%   |
| F1-Score   | 87.31%   |
| ROC AUC    | 93.85%   |

---

## 🚀 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Teena1805/Movie_Review_Sentiment_Analysis.git
cd movie-review-sentiment-analysis

