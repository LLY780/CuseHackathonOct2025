import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


# Config

DATA_PATH = "bias_clean.csv"
TEXT_COL = "page_text"
LABEL_COL = "bias"

VOCAB_SIZE = 30000
MAX_LEN = 500
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
OUTPUT_DIR = Path("./model")


# Helper Functions

def basic_clean(text):
    """Simple normalization used in both training and inference."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def texts_to_padded(texts, tokenizer, max_len):
    """Tokenize + pad. Used for train/val/test."""
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

def main():
    # 1. Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df[TEXT_COL].apply(basic_clean)

    # 2. Encode labels
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df[LABEL_COL])

    X = df["clean_text"].values
    y = df["label_id"].values

    # 3. Split data into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 4. Fit tokenizer on training data
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    # 5. Tokenize and pad data
    X_train_pad = texts_to_padded(X_train, tokenizer, MAX_LEN)
    X_val_pad   = texts_to_padded(X_val,   tokenizer, MAX_LEN)
    X_test_pad  = texts_to_padded(X_test,  tokenizer, MAX_LEN)

    y_train_arr = np.array(y_train)
    y_val_arr   = np.array(y_val)
    y_test_arr  = np.array(y_test)

    # 6. Build model
    num_classes = len(label_encoder.classes_)
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, embeddings_initializer="uniform", trainable=True),
        Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])

    # 7. Train model with early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(X_train_pad, y_train_arr, validation_data=(X_val_pad, y_val_arr), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

    # 8. Evaluate on test data
    test_probs = model.predict(X_test_pad)
    test_pred_ids = np.argmax(test_probs, axis=1)

    test_accuracy = accuracy_score(y_test_arr, test_pred_ids)
    class_report = classification_report(y_test_arr, test_pred_ids, target_names=list(label_encoder.classes_))
    conf_mat = confusion_matrix(y_test_arr, test_pred_ids)

    print("Test accuracy:", test_accuracy)
    print("Classification report:\n", class_report)
    print("Confusion matrix:\n", conf_mat)

    # 9. Save model and preprocessing artifacts
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Keras model
    model.save(OUTPUT_DIR/"model.keras")

    # Tokenizer
    f = open(OUTPUT_DIR/"tokenizer.pkl", "wb")
    pickle.dump(tokenizer, f)

    # Label encoder
    f = open(OUTPUT_DIR/"label_encoder.pkl", "wb")
    pickle.dump(label_encoder, f)

    # Save constants needed at inference
    f = open(OUTPUT_DIR/"constants.pkl", "wb")
    pickle.dump( {"MAX_LEN" : MAX_LEN}, f)


# Run from command line

if __name__ == "__main__":
    main()