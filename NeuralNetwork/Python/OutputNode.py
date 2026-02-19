from HiddenNode import HiddenNode
"""
Output Node:
Final node in the network, responsible for the prediction
of the Y value. Essentially the same as an activation function in a hidden node,
but we dont use the ReLU function, we take the raw output of the equation
as the predicted Y value.
"""

class OutputNode(HiddenNode):

    def __init__(self, bias):
        super().__init__(bias)

    def Prediction(self):
        prediction = self.bias

        for weight in self.incomingWeights:
            prediction += weight.value * weight.prev.value

        self.value = prediction
