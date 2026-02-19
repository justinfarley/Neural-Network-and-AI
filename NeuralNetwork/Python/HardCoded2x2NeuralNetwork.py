from InputNode import InputNode
from HiddenNode import HiddenNode
from OutputNode import OutputNode
from Weight import Weight
from MathUtil import MathUtil

"""
Author: Justin Farley

So I initially made this file to test out my understanding and the math that I implemented
in MathUtil before I created a generic NeuralNetwork.

For reference, this is a 2x2 Neural Network comprised of:
- 2 input nodes (x1, x2)
- 1 hidden layer with 2 hidden nodes (h1, h2)
- 2 output nodes (y1, y2)

Connections:

    x1---h1---y1
      \/   \/    
      /\   /\
    x2---h2---y2
"""

# Assumes Node, HiddenNode, OutputNode, Weight, MathUtil already imported

#Load CSV
Xs = []
Ys = []
learningRate = 0.001
iterations = 100

with open("test2x2.csv", "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(",")
    Xs.append(float(parts[0]))
    Ys.append(float(parts[1]))

# Initialize nodes
x1Node = InputNode(Xs[0])
x2Node = InputNode(Xs[1])

h1 = HiddenNode(0.1)
h2 = HiddenNode(0.2)

y1 = OutputNode(0.3)
y2 = OutputNode(0.5)

# Input → Hidden weights
w11 = Weight(x1Node, h1, 0.5)
w12 = Weight(x1Node, h2, 0.3)
w21 = Weight(x2Node, h1, 0.4)
w22 = Weight(x2Node, h2, 0.6)

# Hidden → Output weights
v11 = Weight(h1, y1, 0.2)
v12 = Weight(h1, y2, 0.5)
v21 = Weight(h2, y1, 0.3)
v22 = Weight(h2, y2, 0.1)

for i in range(iterations):

    # Forward pass
    h1.Activation()
    h2.Activation()
    y1.Prediction()
    y2.Prediction()

    y1.error = MathUtil.GetOutputNodeError(y1.value, Ys[0])
    y2.error = MathUtil.GetOutputNodeError(y2.value, Ys[1])

    predictedValues = [y1.value, y2.value]

    print(f"Iteration: {i+1} FORWARD PASS: "
          f"Predictions: y1={y1.value}, y2={y2.value}")

    # Output gradients
    v11Gradient = MathUtil.GetOutputParameterGradient(y1.error, h1.value)
    v12Gradient = MathUtil.GetOutputParameterGradient(y2.error, h1.value)
    v21Gradient = MathUtil.GetOutputParameterGradient(y1.error, h2.value)
    v22Gradient = MathUtil.GetOutputParameterGradient(y2.error, h2.value)

    # Hidden errors
    hiddenErrors = [h1.HiddenNodeError(), h2.HiddenNodeError()]
    reluDerivatives = [
        1 if h1.value > 0 else 0,
        1 if h2.value > 0 else 0
    ]

    # Input layer gradients
    w11Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[0], reluDerivatives[0], Xs[0])
    w21Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[0], reluDerivatives[0], Xs[1])
    w12Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[1], reluDerivatives[1], Xs[0])
    w22Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[1], reluDerivatives[1], Xs[1])

    biasO1Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[0], reluDerivatives[0], 1)
    biasO2Gradient = MathUtil.GetInputParameterGradient(
        hiddenErrors[1], reluDerivatives[1], 1)

    # Update output weights
    v11.value = MathUtil.UpdateParameter(v11.value, learningRate, v11Gradient)
    v12.value = MathUtil.UpdateParameter(v12.value, learningRate, v12Gradient)
    v21.value = MathUtil.UpdateParameter(v21.value, learningRate, v21Gradient)
    v22.value = MathUtil.UpdateParameter(v22.value, learningRate, v22Gradient)

    # Update input weights
    w11.value = MathUtil.UpdateParameter(w11.value, learningRate, w11Gradient)
    w21.value = MathUtil.UpdateParameter(w21.value, learningRate, w21Gradient)
    w12.value = MathUtil.UpdateParameter(w12.value, learningRate, w12Gradient)
    w22.value = MathUtil.UpdateParameter(w22.value, learningRate, w22Gradient)

    # Update biases
    h1.bias = MathUtil.UpdateParameter(h1.bias, learningRate, biasO1Gradient)
    h2.bias = MathUtil.UpdateParameter(h2.bias, learningRate, biasO2Gradient)
