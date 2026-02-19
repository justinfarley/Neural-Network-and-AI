import math
"""
Just a Util class. Some methods repeat but the naming was helping me 
keep track of the backpropagation process, so I left them as is.
"""

class MathUtil:

    @staticmethod
    def GetOneHotEncodedArr(category, categories):
        vector = [0.0] * len(categories)
        for i in range(len(categories)):
            vector[i] = 1.0 if categories[i] == category else 0.0
        return vector

    @staticmethod
    def MSELoss(predictedValues, actualValues):
        totalLoss = 0.0
        for i in range(len(predictedValues)):
            MSE = (predictedValues[i] - actualValues[i]) ** 2
            totalLoss += MSE

        totalLoss /= len(predictedValues)
        return totalLoss

    @staticmethod
    def CrossEntropyLoss(predicted, actual):
        if len(actual) <= 0:
            return -1

        loss = 0.0
        for i in range(len(predicted)):
            p = max(predicted[i], 1e-15)
            loss -= actual[i] * math.log(p)

        return loss

    @staticmethod
    def GetOutputParameterGradient(error, activationPrevLayer):
        return error * activationPrevLayer

    @staticmethod
    def GetInputParameterGradient(hiddenError, reluDerivative, xValue):
        return hiddenError * reluDerivative * xValue

    @staticmethod
    def GetMiddleLayerParameterGradient(hiddenError, reluDerivative, activationPrevLayer):
        return hiddenError * reluDerivative * activationPrevLayer

    @staticmethod
    def UpdateParameter(oldParameter, learningRate, gradient):
        return oldParameter - learningRate * gradient

    @staticmethod
    def GetOutputNodeError(predictedValue, actualValue):
        return predictedValue - actualValue
