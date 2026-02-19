"""
The Weights!
Weights are the connections between the nodes. 
They have a value, and they also have a reference to the previous node and the next node
of which they are connected.
The value is their bias, which starts random and is adjusted during backpropagation.
"""

class Weight:

    def __init__(self, prev, next, value=0.5):
        self.prev = prev
        self.next = next
        self.value = value

        prev.AddOutgoingWeight(self)
        next.AddIncomingWeight(self)
