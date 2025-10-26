import scrape
import argparse
import pickle
import re
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model + tokenizer + label encoder + constants

OUTPUT_DIR = Path("./model")

model = keras.models.load_model(OUTPUT_DIR/"model.keras")

f = open(OUTPUT_DIR/"tokenizer.pkl", "rb")
tokenizer = pickle.load(f)

f = open(OUTPUT_DIR/"label_encoder.pkl", "rb")
label_encoder = pickle.load(f)

f = open(OUTPUT_DIR/"constants.pkl", "rb")
constants = pickle.load(f)

MAX_LEN = constants["MAX_LEN"]


# Prediction functions

def predict_text(text):
    '''Uses text input to predict bias'''
    # Clean text (same as training)
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to padded sequence
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Predict
    pred = model.predict(pad)
    pred_id = np.argmax(pred, axis=1)[0]
    label = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(np.max(pred))

    print(f"\nPredicted bias: {label} (confidence: {confidence:.2f})")
    print("Probability distribution:")
    for catg, prob in sorted(zip(label_encoder.classes_, pred[0]), key=lambda x: x[1], reverse=True):
        print(f"  {catg}: {prob:.4f}")
    return (label, confidence)

def predict_url(url):
    '''Calls predict_text with url text converted'''
    return predict_text(scrape.getText(url))

def main():
    parser = argparse.ArgumentParser(
        description="Predict bias from text or a URL using a trained CNN model."
    )
    parser.add_argument(
        "input",
        help="Either a text string or a URL (starting with http/https). Wrap multi-word text in quotes."
    )
    args = parser.parse_args()

    user_input = args.input.strip()

    # Determine if input is a URL
    if user_input.startswith("http://") or user_input.startswith("https://"):
        try:
            predict_url(user_input)
        except Exception as e:
            print(f"Error: Unable to fetch text from URL - {e}")
            return
    else:
        predict_text(user_input)


# Run from command line

if __name__ == "__main__":
    main()