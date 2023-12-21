import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Define the RNN model
class RNNModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn_layer = tf.keras.layers.SimpleRNN(
            hidden_size, return_sequences=True, return_state=True
        )
        self.dense_layer = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        output, _ = self.rnn_layer(inputs)
        return self.dense_layer(output)


# Parameters
input_size = 100
hidden_size = 64
output_size = 1  #

# Create the model instance
model = RNNModel(input_size, hidden_size, output_size)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Load the data
with open('./OutputSplit/final_out.txt', 'r') as file:
    data = np.array([float(line.strip()) for line in file])

# Preprocess the data
X = []
y = []
for i in range(len(data) - input_size - 1):
    X.append(data[i:i+input_size])
    y.append(data[i+input_size])

X = np.array(X).reshape(-1, input_size, 1)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Load and preprocess the test data
def load_and_preprocess(file_path, input_size):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file])

    X = []
    for i in range(len(data) - input_size - 1):
        X.append(data[i:i+input_size])

    return np.array(X).reshape(-1, input_size, 1)

# Load test data
X_test = load_and_preprocess('./dataset/val.txt', input_size)

# Predict using the model
y_test_pred = model.predict(X_test)
print(y_test_pred)