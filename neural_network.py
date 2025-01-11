import numpy as np
    
# Goal: Get signal 1 for buy, -1 for sell and 0 for do nothing
class NeuralNetwork:
    def __init__(self, input_size=768, hidden_layers=[512, 512], output_size=1):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Creates a (input_size by hidden_layers[0]) matrix, 768 x 512 unless otherwise specified
        # Each row is a set of weights for one neuron
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros(1, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[0], hidden_layers[i + 1]))
            self.biases.append(np.zeros(1, hidden_layers[i + 1]))

        # Last hidden network to output
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros(1, output_size))

    def forward(self, inputs):
        # layers contains all activations
        layers = [inputs]

        for i in range(len(self.weights)):
            layers.append(np.dot(layers[-1], self.weights[i]) + self.biases[i])

        return layers[-1]
    