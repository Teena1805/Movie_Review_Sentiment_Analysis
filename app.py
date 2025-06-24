from flask import Flask, render_template, request
import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tools
model = load_model('model/lstm_sentiment_model.h5')
tokenizer = joblib.load('model/tokenizer.joblib')
label_encoder = joblib.load('model/label_encoder.joblib')
max_len = 200

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)         # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)       # Remove non-letter characters
    text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
    return text

# 🏠 Home Page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# 🔍 Sentiment Prediction
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned_review])
    padded = pad_sequences(seq, maxlen=max_len)
    
    pred_prob = model.predict(padded)[0][0]
    sentiment_label = "positive" if pred_prob >= 0.5 else "negative"
    confidence = pred_prob if pred_prob >= 0.5 else 1 - pred_prob

    return render_template('index.html',
                           review=review,
                           sentiment=sentiment_label,
                           confidence=f"{confidence*100:.2f}%")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)