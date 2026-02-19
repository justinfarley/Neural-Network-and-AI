"""
This is the base node class. 
It shows the basic properties that all 3 types of nodes (input, hidden, and output)
all have. 
Each node has a value, whether that be an x value, an activation value, or an output value.
They also have an error, and lists of incoming and outgoing weights. 
(this may not be optimal idk)
"""

class Node:
    # base class for input and hidden nodes

    def __init__(self):
        self.value = 0.0
        self.error = 0.0

        self.outgoingWeights = []
        self.incomingWeights = []

    def AddOutgoingWeight(self, weight):
        self.outgoingWeights.append(weight)

    def AddIncomingWeight(self, weight):
        self.incomingWeights.append(weight)
