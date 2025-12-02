import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MAX_WORDS = 5000
MAX_LEN = 30

_model = None
_tokenizer = None


def load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    # Kaggle datasını oku
    df = pd.read_csv("data/sentences.csv")
    texts = df["text"].astype(str).tolist()

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    _tokenizer = tokenizer
    return tokenizer


def load_emotion_model():
    global _model
    if _model is not None:
        return _model

    model = load_model("emotion_model.keras")
    _model = model
    return model


def predict_emotion(text: str) -> str:
    tokenizer = load_tokenizer()
    model = load_emotion_model()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prob = model.predict(padded)[0][0]
    if prob >= 0.5:
        label = "sad"
    else:
        label = "happy"

    return label
