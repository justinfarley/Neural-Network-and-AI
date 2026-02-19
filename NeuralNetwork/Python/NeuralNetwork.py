import math
import random
import time
from enum import Enum
from InputNode import InputNode
from HiddenNode import HiddenNode
from OutputNode import OutputNode
from Weight import Weight
from MathUtil import MathUtil
"""
Author: Justin Farley

This was fun to build
"""

class NeuralNetwork:

    class PredictionMethod(Enum):
        Linear = 0
        Softmax = 1

    def __init__(self, learningRate, inputSize, outputSize, hiddenLayers, hiddenNodesPerLayer):
        self.learningRate = learningRate
        self.hiddenLayers = hiddenLayers
        self.hiddenNodesPerLayer = hiddenNodesPerLayer
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.inputNodes = []
        self.outputNodes = []
        self.hiddenNodes = {}  # dict[int, list[HiddenNode]]
        self.weights = []

    def GetAllWeights(self):
        return self.weights

    def GetHiddenLayers(self):
        return self.hiddenNodes

    def GetOutputNodes(self):
        return self.outputNodes

    def GetInputNodes(self):
        return self.inputNodes

    def GetNodeIndex(self, node):
        index = 0

        for i in range(len(self.inputNodes)):
            if self.inputNodes[i] == node:
                return index
            index += 1

        for key in sorted(self.hiddenNodes.keys()):
            for hiddenNode in self.hiddenNodes[key]:
                if hiddenNode == node:
                    return index
                index += 1

        for i in range(len(self.outputNodes)):
            if self.outputNodes[i] == node:
                return index
            index += 1

        return -1

    @staticmethod
    def GetWeightValue(prevLayerCount):
        fanIn = prevLayerCount
        scale = math.sqrt(2.0 / fanIn)
        weightValue = (random.random() * 2.0 - 1.0) * scale
        return weightValue

    def NetworkInit(self):

        # Input nodes
        for i in range(self.inputSize):
            inputNode = InputNode(0)
            self.inputNodes.append(inputNode)

        # First hidden layer
        self.hiddenNodes[0] = []
        for j in range(self.hiddenNodesPerLayer):
            hiddenNode = HiddenNode(self.GetWeightValue(len(self.inputNodes)))
            self.hiddenNodes[0].append(hiddenNode)

        for hiddenNode in self.hiddenNodes[0]:
            for inputNode in self.inputNodes:
                weight = Weight(inputNode, hiddenNode,
                                self.GetWeightValue(len(self.inputNodes)))
                self.weights.append(weight)

        # Remaining hidden layers
        for i in range(1, self.hiddenLayers):
            self.hiddenNodes[i] = []
            for j in range(self.hiddenNodesPerLayer):
                hiddenNode = HiddenNode(self.GetWeightValue(self.hiddenNodesPerLayer))
                self.hiddenNodes[i].append(hiddenNode)

        for key, layer in self.hiddenNodes.items():
            for hiddenNode in layer:
                if key == 0:
                    continue
                for prevHiddenNode in self.hiddenNodes[key - 1]:
                    weight = Weight(prevHiddenNode, hiddenNode,
                                    self.GetWeightValue(self.hiddenNodesPerLayer))
                    self.weights.append(weight)

        # Output nodes
        for i in range(self.outputSize):
            outputNode = OutputNode(random.random())
            self.outputNodes.append(outputNode)

            for hiddenNode in self.hiddenNodes[self.hiddenLayers - 1]:
                weight = Weight(hiddenNode, outputNode,
                                self.GetWeightValue(self.hiddenNodesPerLayer))
                self.weights.append(weight)

    def ForwardPass(self, input, method=PredictionMethod.Linear, expected=None):

        for i in range(len(self.inputNodes)):
            self.inputNodes[i].value = input[i]

        for key in self.hiddenNodes:
            for hiddenNode in self.hiddenNodes[key]:
                hiddenNode.Activation()

        predictedValues = []

        for j in range(len(self.outputNodes)):
            outputNode = self.outputNodes[j]
            outputNode.Prediction()

            if expected is not None and method == self.PredictionMethod.Linear:
                outputNode.error = MathUtil.GetOutputNodeError(outputNode.value, expected[j])

            predictedValues.append(outputNode.value)

        if method != self.PredictionMethod.Softmax and expected is not None:
            print(f"Loss (MSE): {MathUtil.MSELoss(expected, predictedValues)}")

        if method == self.PredictionMethod.Softmax:

            predictedValues = []
            rawOutputs = [x.value for x in self.outputNodes]

            maxVal = max(rawOutputs)
            sums = sum(math.exp(x - maxVal) for x in rawOutputs)

            for i in range(len(self.outputNodes)):
                softmaxVal = math.exp(rawOutputs[i] - maxVal) / sums
                self.outputNodes[i].value = softmaxVal
                predictedValues.append(softmaxVal)

                if expected is not None:
                    self.outputNodes[i].error = softmaxVal - expected[i]

        return predictedValues

    def BackwardsPass(self):

        weightGradients = {}
        biasGradients = {}

        # Output gradients
        for outputNode in self.outputNodes:
            for weight in outputNode.incomingWeights:
                gradient = MathUtil.GetOutputParameterGradient(
                    outputNode.error, weight.prev.value)
                weightGradients[weight] = gradient

            biasGradients[outputNode] = MathUtil.GetOutputParameterGradient(
                outputNode.error, 1.0)

        # Hidden error calculation
        for j in range(self.hiddenLayers - 1, -1, -1):
            for hiddenNode in self.hiddenNodes[j]:
                error = 0
                for outgoingWeight in hiddenNode.outgoingWeights:
                    error += outgoingWeight.next.error * outgoingWeight.value

                hiddenNode.error = error * (1.0 if hiddenNode.value > 0 else 0.0)

        # Hidden layer gradients
        for j in range(self.hiddenLayers - 1, 0, -1):
            for hiddenNode in self.hiddenNodes[j]:
                for weight in hiddenNode.incomingWeights:
                    gradient = hiddenNode.error * weight.prev.value
                    weightGradients[weight] = gradient

                biasGradients[hiddenNode] = hiddenNode.error

        # Input layer gradients
        for hiddenNode in self.hiddenNodes[0]:
            for weight in hiddenNode.incomingWeights:
                gradient = hiddenNode.error * weight.prev.value
                weightGradients[weight] = gradient

            biasGradients[hiddenNode] = hiddenNode.error

        # Update weights
        for weight in self.weights:
            weight.value = MathUtil.UpdateParameter(
                weight.value,
                self.learningRate,
                weightGradients[weight]
            )

        # Update hidden biases
        for key in self.hiddenNodes:
            for hiddenNode in self.hiddenNodes[key]:
                hiddenNode.bias = MathUtil.UpdateParameter(
                    hiddenNode.bias,
                    self.learningRate,
                    biasGradients[hiddenNode]
                )

        # Update output biases
        for outputNode in self.outputNodes:
            outputNode.bias = MathUtil.UpdateParameter(
                outputNode.bias,
                self.learningRate,
                biasGradients[outputNode]
            )

        weightGradients.clear()
        biasGradients.clear()

    def Predict(self, input, method=PredictionMethod.Linear):
        return self.ForwardPass(input, method)

    def Train(self, inputs, expected, iterations,
              method=PredictionMethod.Linear):

        print(f"Began Training @ {time.strftime('%H:%M:%S')}")

        for j in range(iterations):

            start = time.time()
            totalLoss = 0

            for i in range(len(inputs)):
                preds = self.ForwardPass(inputs[i], method, expected[i])
                self.BackwardsPass()
                totalLoss += MathUtil.CrossEntropyLoss(preds, expected[i])

            avgLoss = totalLoss / len(inputs)
            elapsed = int(time.time() - start)

            print(f"Epoch {j+1}/{iterations} - Avg Loss: {avgLoss}, "
                  f"Time for this epoch: {elapsed//60}m and {elapsed%60}s "
                  f"@ {time.strftime('%H:%M:%S')}")

    def TrainSingle(self, inputs, expected, iterations,
                    method=PredictionMethod.Linear):

        for i in range(iterations):
            self.ForwardPass(inputs, method, expected)
            self.BackwardsPass()
