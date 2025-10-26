TF_ENABLE_ONEDNN_OPTS=0
import pickle
import numpy as np
import keras
from keras import ops

import pandas as pd
fileName = "../../Downloads/bias_clean.csv"

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reload everything (if running in new session)
model = load_model("model 1.keras")
with open("../../Downloads/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("../../Downloads/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Text cleaner (same one you used earlier)
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def predict_bias(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=400, padding='post', truncating='post')
    preds = model.predict(pad)
    class_idx = np.argmax(preds, axis=1)[0]
    label = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(preds))
    print(f"\nPredicted bias: {label} (confidence: {confidence:.2f})")
    print("Probability distribution:")
    for lbl, prob in zip(label_encoder.classes_, preds[0]):
        print(f"  {lbl:>10}: {prob:.3f}")
    return label
df = pd.read_csv(fileName)
df.drop(index=df.index[0])
def readText(x): #int of the row starting at 1
    if df.loc[x-1, "page_text"] == "Error fetching article":
        return "empty"
    return df.loc[x-1,"page_text"]
import Scrapper as sc
# e.g "https://www.cnn.com/2025/10/25/business/trump-tariffs-canada-reagan"
url = input("What\'s your url?: ")
predict_bias(sc.getText(url))