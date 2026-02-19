from Node import Node
"""
The Hidden Node. Used in the hidden layers of the neural network
between the input and output layers. Each node has an activation function which is 
w1x1 + w2x2 + ... + b, where w is the weight of the connection, 
x is the x value, and b is the bias. 
The equation is then passed through a ReLU function to see if the neuron "fires" or not.
"""

class HiddenNode(Node):

    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    # z1 = w1x1 + w2x2 + b
    def Activation(self):
        activation = self.bias

        for weight in self.incomingWeights:
            activation += weight.value * weight.prev.value

        activation = max(0, activation)  # ReLU activation function

        self.value = activation

    def HiddenNodeError(self):
        error = 0.0
        for weight in self.outgoingWeights:
            error += weight.value * weight.next.error

        self.error = error
        return error
