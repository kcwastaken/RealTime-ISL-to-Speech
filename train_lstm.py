import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

ACTIONS = np.array(['Namaste', 'ThankYou', 'Sorry', 'Hello', 'Busy', 'Clean'])

# 1. SETUP PATHS
DATA_PATH = os.path.join('MP_Data') 
NO_SEQUENCES = 30 
SEQUENCE_LENGTH = 30 

# 2. LOAD DATA 
label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        window = []
        skip_sequence = False
        
        for frame_num in range(SEQUENCE_LENGTH):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if os.path.exists(file_path):
                res = np.load(file_path)
                window.append(res)
            else:
                skip_sequence = True
                break 
        
        if not skip_sequence:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels, num_classes=ACTIONS.shape[0]).astype(int)
print(f"Total sequences loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. BUILD THE ADVANCED LSTM MODEL
print("Building the AI architecture...")
model = Sequential()
# Notice the 126 right here!
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

# 4. COMPILE AND TRAIN
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Training started... this will take a minute or two.")
model.fit(X_train, y_train, epochs=200) 

# 5. SAVE THE MODEL
model.save('action.h5')
print("Model trained and saved successfully as action.h5!")