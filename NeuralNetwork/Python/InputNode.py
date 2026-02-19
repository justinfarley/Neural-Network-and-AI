from Node import Node
"""
Input node:
pretty straight forward, just holds the x value.
"""

class InputNode(Node):

    def __init__(self, xValue):
        super().__init__()
        self.value = xValue
