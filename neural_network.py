import numpy as np
    
# Goal: Get a signal for buy, hold or sell
class NeuralNetwork:
    def __init__(self, input_size=768, hidden_layers=[512, 512], output_size=3):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Creates a (input_size by hidden_layers[0]) matrix, 768 x 512 unless otherwise specified
        # Each row is a set of weights for one neuron
        self.weights.append(0.0001 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[0], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Last hidden network to output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    # activation function
    def tanh(self, activation):
        return np.tanh(activation)
        
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

 
    def forward(self, inputs):
        # layers contains all activations
        self.activations = [inputs.reshape(1, -1)]

        for i in range(len(self.weights)):
            x = np.array(self.activations[-1].reshape(1, -1))
            w = np.array(self.weights[i])

            z = np.dot(x, w) + self.biases[i]
            a = self.tanh(z)
            self.activations.append(a)

        return self.activations
    
    # Calculated using chainrule, doutput/dw = dz/dw*da/dz*doutput/da
    def backward(self, targets: np.ndarray):
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # error = MSE, therefore doutput/da = 2 * error
        delta = 2 * (self.activations[-1] - targets)
        # da / dz
        delta *= np.vectorize(self.tanh_derivative)(self.activations[-1])

        for i in reversed(range(len(self.weights))):
            # dz / dw
            weight_gradients[i] = np.dot(self.activations[i].T, delta)
            # dz / da
            bias_gradients[i] = np.sum(delta, axis=0)

            if i > 0:  # Skip for the input layer
                delta = np.dot(delta, self.weights[i].T) * np.vectorize(self.tanh_derivative)(self.activations[i])

        return weight_gradients, bias_gradients

    def backpropagation(self, minibatch: list, targets: np.ndarray, learning_rate: float):
        total_weight_gradients = [np.zeros_like(w) for w in self.weights]
        total_bias_gradients = [np.zeros_like(b) for b in self.biases]


        for i in range(len(minibatch)):
            self.forward(minibatch[i])

            weight_gradients, bias_gradients = self.backward(targets[i])

            for j in range(len(self.weights)):
                total_weight_gradients[j] += weight_gradients[j]
                total_bias_gradients[j] += bias_gradients[j]

        num_examples = len(minibatch)
        averaged_weight_gradients = [wg / num_examples for wg in total_weight_gradients]
        averaged_bias_gradients = [bg / num_examples for bg in total_bias_gradients]

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * averaged_weight_gradients[i]
            self.biases[i] -= learning_rate * averaged_bias_gradients[i]
        
    def train(self, data, targets, epochs=10, batch_size=32):
        num_examples = len(data)
        for epoch in range(epochs):
            permutation = np.random.permutation(num_examples)
            shuffled_data = data[permutation]
            shuffled_targets = targets[permutation]
            
            for i in range(0, num_examples, batch_size):
                minibatch_data = shuffled_data[i:i + batch_size]
                minibatch_targets = shuffled_targets[i:i + batch_size]

                # Hardcoding learning rate
                # Would be cool with changing learning rate after how steep the gradient is
                self.backpropagation(minibatch_data, minibatch_targets, learning_rate=0.0001)

            # print(f"targets: {targets}")
            # print(f"targets[epoch]: {targets[epoch]}")
            # if epoch == 0 or epoch == epochs:

            prediction = self.predict(data[epoch])

            accuracy = self.evaluate(prediction, targets[epoch])
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}\n\n")

    def predict(self, data):
        return self.forward(data)[-1]

    def evaluate(self, predictions, targets):
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)

        print(f"predictions: {predictions}")
        print(f"targets: {targets}\n\n")

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)

        grades = np.where(predicted_classes == true_classes, 1, 0)
        accuracy = np.mean(grades)
        return accuracy


def create_targets(data, margin_per_share=20, horizon=14):
    window_size = horizon * 10
    num_targets = len(data["Close"]) - horizon
    input_size = window_size + 1  # Includes the current price
    training_data = np.zeros((num_targets - window_size, input_size))
    targets = np.zeros((num_targets - window_size, 3))  # [buy, hold, sell]

    for i in range(window_size, num_targets):
        training_window = data["Close"].iloc[i - window_size:i + 1].values.astype(float)
        training_data[i - window_size] = training_window

        current_price = float(data["Close"].iloc[i]) 
        future_price = float(data["Close"].iloc[i + horizon])

        # Calculate target based on price movement
        if future_price >= current_price + margin_per_share:
            targets[i - window_size] = np.array([1, 0, 0])  # Buy
        elif future_price <= current_price - margin_per_share:
            targets[i  - window_size] = np.array([0, 0, 1])  # Sell
        else:
            targets[i  - window_size] = np.array([0, 1, 0])  # Hold


    return np.array(training_data), targets