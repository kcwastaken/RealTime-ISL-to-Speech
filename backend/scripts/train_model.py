import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = os.path.join('data', 'mp_data')
actions = np.array(["Namaste", "Help", "Doctor", "Medicine", "Water", "Hello", "MyNameIs", "Krishna", "Sorry", "Neutral"])
no_sequences = 30
sequence_length = 30

print("Loading data...")
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("Building neural network...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 144)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Training started! This might take a few minutes...")
model.fit(X_train, y_train, epochs=200)

save_path = os.path.join('backend', 'models', 'action_v4.h5')
model.save(save_path)
print(f"Training Complete! New model saved as '{save_path}'")