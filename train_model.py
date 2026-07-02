import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Setup the full vocabulary (MUST match the exact folder names in MP_Data)
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Namaste', 'ThankYou', 'Sorry', 'Hello', 'Busy', 'Clean', 'MyNameIs', 'Krishna'])
no_sequences = 30
sequence_length = 30

print("Loading data...")
# 2. Load all the .npy files into memory
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

# 3. Prepare Data for Training
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("Building neural network...")
# 4. Build the LSTM Architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# CRITICAL CHANGE: The output layer is now dynamic based on how many words we have (8)
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Training started! This might take a few minutes...")
# 5. Train the Model! 
# (You can increase epochs to 1000 or 2000 later if accuracy is low)
model.fit(X_train, y_train, epochs=500)

# 6. Save the new brain!
model.save('action_v2.h5')
print("Training Complete! New model saved as 'action_v2.h5'")