import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "mp_data"
MODEL_DIR = BASE_DIR / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

actions = np.array(["Namaste", "Help", "Doctor", "Medicine", "Water", "Hello", "MyNameIs", "Krishna", "Sorry", "Neutral"])
sequence_length = 30

# How many sequences exist per action under DATA_PATH. Bumping this (and
# collecting more data with collect_data.py first) is the single highest-
# leverage accuracy fix available right now - 30 sequences/class is thin
# for a model this size. Detected automatically so you don't have to keep
# this number in sync by hand.
no_sequences = min(
    len(list((DATA_PATH / action).iterdir())) for action in actions
)
print(f"[INFO] Using {no_sequences} sequences per action (auto-detected).")

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(DATA_PATH / action / str(sequence) / f"{frame_num}.npy")
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y_labels = np.array(labels)
y = to_categorical(y_labels).astype(int)

# stratify=y_labels keeps every class represented proportionally in the
# validation split. With only ~30 sequences/class, an unstratified split
# could easily leave a class with 0-1 validation samples, making val_loss
# (which EarlyStopping watches) unreliable.
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X, y, y_labels, test_size=0.15, random_state=42, stratify=y_labels
)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="tanh", input_shape=(30, 144)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

save_path = MODEL_DIR / "action_v4.h5"
model.save(str(save_path))
print(f"[SUCCESS] Model saved to {save_path}")

# --- Per-class breakdown -------------------------------------------------
# Aggregate val_categorical_accuracy hides which specific signs are weak.
# This shows exactly that, so threshold/data-collection effort can go to
# the classes that actually need it instead of guessing.
val_pred_probs = model.predict(X_test)
val_pred_labels = np.argmax(val_pred_probs, axis=1)

print("\n[INFO] Per-class validation report:")
print(classification_report(labels_test, val_pred_labels, target_names=actions))

print("[INFO] Confusion matrix (rows = true, cols = predicted):")
cm = confusion_matrix(labels_test, val_pred_labels, labels=list(range(len(actions))))
header = "        " + " ".join(f"{a[:6]:>6}" for a in actions)
print(header)
for i, row in enumerate(cm):
    print(f"{actions[i][:6]:>6}  " + " ".join(f"{v:>6}" for v in row))