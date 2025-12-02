import pandas as pd
import numpy as np
import pickle  # <-- eklendi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# CSV yükle
df = pd.read_csv("data/sentences.csv")

# Text ve Label ayır
texts = df["text"].values
labels = df["label"].values

# Label encode
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize et
MAX_NUM_WORDS = 5000
MAX_LEN = 30

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN)

# Train / Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels_encoded, test_size=0.2, random_state=42
)

# Model
model = Sequential([
    Embedding(MAX_NUM_WORDS, 64),
    LSTM(64, return_sequences=False),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Eğit
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli kaydet
model.save("emotion_model.keras")
print("Model kaydedildi!")

# Tokenizer'ı kaydet
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer kaydedildi!")
