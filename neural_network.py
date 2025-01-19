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
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[0], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Last hidden network to output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    # activation function
    def relu(self, activation):
        return np.maximum(0, activation)
    
    def relu_derivative(x):
        return 1 if x > 0 else 0
 
    def forward(self, inputs):
        # layers contains all activations
        self.activations = [inputs]

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)

        return self.activations
    
    # Calculated using chainrule, doutput/dw = dz/dw*da/dz*doutput/da
    def backward(self, targets: np.ndarray):
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # error = MSE, therefore doutput/da = 2 * error
        delta = 2 * (self.activations[-1] - targets)
        # da / dz
        delta *= np.vectorize(self.relu_derivative)(self.activations[-1])

        for i in reversed(range(len(self.weights))):
            # dz / dw
            weight_gradients[i] = np.dot(self.activations[i].T, delta)
            # dz / da
            bias_gradients[i] = np.sum(delta, axis=0)

            if i > 0:  # Skip for the input layer
                delta = np.dot(delta, self.weights[i].T) * np.vectorize(self.relu_derivative)(self.activations[i])

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
        # The real challenge is to evaluate if the market really is bullish or bearish
        num_examples = len(data)
        for epoch in range(epochs):
            permutation = np.random.permutation(num_examples)
            shuffled_data = data[permutation]
            shuffled_targets = targets[permutation]
            
            for i in range(0, num_examples, batch_size):
                minibatch_data = shuffled_data[i:i + batch_size]
                minibatch_targets = shuffled_targets[i:i + batch_size]

                # Hardcoding learning rate to
                # Would be cool with changing learning rate after how steep the gradient is
                self.backpropagation(minibatch_data, minibatch_targets, learning_rate=0.01)

            predictions = self.predict(data)
            accuracy = self.evalute(predictions, targets)
            print(f"Epoch {epoch}/{epochs}, Accuracy: {accuracy}")

    def predict(self, data):
        return self.forward(data)[-1]

    def evaluate(self, predictions, targets):
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)

        grades = np.where(predicted_classes == true_classes, 1, 0)
        accuracy = np.mean(grades)
        return accuracy
