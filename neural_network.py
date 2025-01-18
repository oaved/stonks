import numpy as np
    
# Goal: Get a signal for buy, hold or sell
class NeuralNetwork:
    def __init__(self, input_size=768, hidden_layers=[512, 512], output_size=3, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
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

        return self.activations[-1]
    
    def backward(self, output: np.ndarray, actual_values: np.ndarray):
        loss = (output - actual_values)**2
        gradients = 2 * [loss * self.relu_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            gradient = 2 * np.dot(gradients[0], self.weights[i].T) * self.relu_derivative(self.activations[i])
            gradients.insert(0, gradient)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, gradients[i])
            self.biases[i] -= self.learning_rate * np.sum(gradients[i], axis=0, keepdims=True)

        

def test():
    inputs = np.random.randn(1, 768)
    myNeuralNetwork = NeuralNetwork(input_size=inputs.shape[1])
    result = myNeuralNetwork.forward(inputs)
    print(f"Here is the result: {result}")

test()