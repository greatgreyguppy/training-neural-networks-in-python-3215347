import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights of the perceptron to the values in w_init."""
        if len(w_init) != len(self.weights):
            raise ValueError("The number of weights must match the number of inputs.")
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
