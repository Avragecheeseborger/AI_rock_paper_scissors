import os, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Data
data = ['\u2702\ufe0f', '\ud83d\udddc\ufe0f', '\ud83d\udddc\ufe0f', '\ud83d\udddc\ufe0f', '\ud83e\udea8', '\ud83e\udea8', '\u2702\ufe0f', '\ud83e\udea8', '\ud83e\udea8', '\u2702\ufe0f', '\u2702\ufe0f', '\ud83d\udddc\ufe0f', '\u2702\ufe0f', '\ud83e\udea8', '\u2702\ufe0f', '\ud83d\udddc\ufe0f', '\ud83e\udea8', '\ud83d\udddc\ufe0f', '\u2702\ufe0f', '\ud83e\udea8', '\u2702\ufe0f']

# Convert symbols into numerical values
tocans = []
for x in data:
    if x == '\u2702\ufe0f':
        tocans.append(0)  # Scissors
    elif x == '\ud83d\udddc\ufe0f':
        tocans.append(1)  # Paper
    elif x == '\ud83e\udea8':
        tocans.append(2)  # Rock

tocans = np.array(tocans)

# Prepare Input Data: We'll use simple one-hot encoding for these 3 possible values
input_data = []
for x in data:
    if x == '\u2702\ufe0f':
        input_data.append([1, 0, 0])  # Scissors
    elif x == '\ud83d\udddc\ufe0f':
        input_data.append([0, 1, 0])  # Paper
    elif x == '\ud83e\udea8':
        input_data.append([0, 0, 1])  # Rock

input_data = np.array(input_data)

# Create the model
model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes (Scissors, Paper, Rock)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save weights as a text string
def get_weights_as_text(model):
    weights = model.get_weights()
    weights_as_lists = [w.tolist() for w in weights]
    return str(weights_as_lists)

# Load weights from a text string
def set_weights_from_text(model, weights_text):
    weights_as_lists = eval(weights_text)  # Convert string back to list
    weights = [np.array(w) for w in weights_as_lists]
    model.set_weights(weights)

# Ask if loading from a previous session
load_session = input("Load from previous session? (y/n): ").strip().lower()
if load_session == 'y':
    weights_text = input("Paste the weights text here: ").strip()
    set_weights_from_text(model, weights_text)
else:
    # Train the model if no weights are provided
    model.fit(input_data, tocans, epochs=100)

# Game Loop
def predict_symbol():
    random_input = np.random.randint(3)
    input_vector = np.zeros((1, 3))
    input_vector[0, random_input] = 1
    prediction = model.predict(input_vector, verbose=0)
    class_index = np.argmax(prediction)
    return class_index, prediction

while True:
    try:
        input_move = input("rock, paper, or scissors: ").strip().lower()

        if input_move not in ['rock', 'paper', 'scissors']:
            print("Invalid input. Exiting.")

            # Print the weights to copy and paste for future sessions
            print("\nCopy and save the following weights text for your next session:")
            print(get_weights_as_text(model))
            break

        predicted_class, probabilities = predict_symbol()
        class_to_symbol = {0: 'scissors', 1: 'paper', 2: 'rock'}
        predicted_symbol = class_to_symbol[predicted_class]

        probabilities2 = []

        for x in probabilities[0]:
            probabilities2.append(math.floor(x*10)/10)

        print(f"Your move: {input_move}")
        print(f"Predicted Move: {predicted_symbol}")
        print(f"Probabilities: {probabilities2}")

        if input_move == predicted_symbol:
            print("It's a tie!")
        elif (input_move == 'rock' and predicted_symbol == 'scissors') or \
            (input_move == 'paper' and predicted_symbol == 'rock') or \
            (input_move == 'scissors' and predicted_symbol == 'paper'):
            print("You win!\n")
        else:
            print(f"You lose! {predicted_symbol} beats {input_move}.")

        # Update data and retrain the model based on the player's move
        if input_move == 'scissors':
            tocans = np.append(tocans, 0)
            input_data = np.append(input_data, [[1, 0, 0]], axis=0)
        elif input_move == 'paper':
            tocans = np.append(tocans, 1)
            input_data = np.append(input_data, [[0, 1, 0]], axis=0)
        elif input_move == 'rock':
            tocans = np.append(tocans, 2)
            input_data = np.append(input_data, [[0, 0, 1]], axis=0)

        model.fit(input_data, tocans, epochs=10, verbose=0)

    except KeyboardInterrupt:
        # Print the weights to copy and paste for future sessions
        print("\nCopy and save the following weights text for your next session:")
        print(get_weights_as_text(model))
        print("Game interrupted. Goodbye!")
        break
