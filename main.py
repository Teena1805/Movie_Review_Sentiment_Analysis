import pandas as pd
import numpy as np
import re
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 1. Load dataset (CSV file with 'review' and 'sentiment' columns)
data = pd.read_csv('data/IMDB_Dataset.csv')

# 2. Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['review'] = data['review'].apply(clean_text)

# 3. Encode labels: positive=1, negative=0
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# 4. Tokenization and padding
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['review'])
sequences = tokenizer.texts_to_sequences(data['review'])
X = pad_sequences(sequences, maxlen=max_len)

y = data['sentiment'].values

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 7. Train model with EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 8. Evaluate model
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

# 9. Save model and tokenizer
model.save('model/lstm_sentiment_model.h5')
joblib.dump(tokenizer, 'model/tokenizer.joblib')
joblib.dump(le, 'model/label_encoder.joblib')

print("Model and tokenizer saved!")
