# 🎬 Movie Review Sentiment Analysis

A Deep Learning-based web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**. The system uses an **LSTM model** trained on 50,000 labeled reviews from the IMDB dataset, with a simple Flask-based frontend for real-time analysis.


## Objective

The goal of this project is to automatically detect the **sentiment of movie reviews** using **Natural Language Processing (NLP)** and **Deep Learning**, making it possible to automate tasks like:

- Classifying user feedback
- Moderating review platforms
- Monitoring social media or comment sections


## 📌 Features

- ✅ Accepts custom movie review text
- ✅ Predicts sentiment (Positive / Negative)
- ✅ Displays prediction confidence
- ✅ Preprocessed with Tokenizer & Padding
- ✅ Trained LSTM model using TensorFlow & Keras
- ✅ Accuracy ~87.1%, Precision ~86.7%, Recall ~87.8%, F1-score ~87.3%, ROC AUC ~93.8%
- ✅ Simple and interactive web interface using Flask


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


# Getting Started
## Prerequisites
Make sure the following are installed:
- Python 3.10+
- pip (Python package manager)
- Git
- Virtual Environment(recommended)


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

## 🚀 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Teena1805/Movie_Review_Sentiment_Analysis.git
cd movie-review-sentiment-analysis
```
---
### 2. **(Optional) Create and activate a virtual environment:**

For **macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

For **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```
---
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is not available, install manually:
```bash
pip install flask tensorflow pandas numpy scikit-learn joblib
```
---
### Step4: Train the Model (Optional, if not using the existing model)
```bash
python main.py
```
This creates:
- model/lstm_sentiment_model.h5
- model/tokenizer.joblib
- model/label_encoder.joblib
- ---
### Step 5: Run the Web App
```bash
python app.py
```
---
### Step6: Now, open your browser and go to:
```bash
http://127.0.0.1:5000/
```
---
## 🧪 Example Results

Here’s a sample prediction made by the model:

**Input Review**:
> "I just can't hold my tears after watching the climax scene. Truly emotional!"

**✅ Predicted Sentiment**: `Positive`  

**🎯 Confidence Score**: `91.47%`

The model effectively captures emotional and sentimental tones in real reviews.


## 🎥 Demo Video

👉 [Watch Demo](https://drive.google.com/file/d/1NQMG2gWGxaEhVGaWIMsSuAcD-ED6rWad/view?usp=drive_link)


## 🧠 Model Architecture

- **Model Type**: LSTM (Long Short-Term Memory)
- **Embedding Layer**: Converts words to vectors
- **LSTM Layer**: Captures temporal relationships in word sequences
- **Dense Layers**: Final classification

This architecture is best for sentiment analysis because it handles long sequences and preserves contextual meaning over time.


## 📈 Model Performance

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 87.13%    |
| Precision    | 86.74%    |
| Recall       | 87.89%    |
| F1 Score     | 87.31%    |
| ROC AUC      | 93.85%    |

These scores are based on a dataset of 50,000 IMDB reviews.


## 🛠 Future Improvements

- Implement BERT or Transformer-based models for improved accuracy
- Add language translation for multilingual review analysis
- Include a dashboard for visualizing sentiment trends
- Deploy the model using Docker and host on cloud (e.g., AWS, Heroku)
- 

## 💬 Feedback & Contributing

Feel free to:
- 🌟 Star this repo if you find it useful
- 🛠 Open issues if you spot bugs or have suggestions
- 📩 Submit a pull request if you'd like to contribute

Your support is appreciated!


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




