# Author: Justin Farley
# 
# This file was my initial understanding of basic Gradient Descent 
# before tackling neural networks themselves.
# Since this gradient descent weight update logic is used in neural networks
# this was a good starting point. It was also very cool to see it "learn" the equation
# (it was even cooler seeing it learn the values themselves <see cref="Neural_Network_and_AI.NeuralNetwork"/>)

import csv

# Read data
Xs = []
Ys = []

learningRate = 0.0000001
iterations = 2000000

theta0 = 0.0
theta1 = 0.0


def linearModel(theta1, X, theta0):
    return theta1 * X + theta0


# Load CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        Xs.append(float(row[0]))
        Ys.append(float(row[1]))


def GetGradientTheta0():
    gradientValueTheta0 = 0.0
    for i in range(len(Xs)):
        gradientValueTheta0 += (
            linearModel(theta1, Xs[i], theta0) - Ys[i]
        )

    gradientValueTheta0 /= len(Xs)
    return gradientValueTheta0


def GetGradientTheta1():
    gradientValueTheta1 = 0.0
    for i in range(len(Xs)):
        gradientValueTheta1 += (
            (linearModel(theta1, Xs[i], theta0) - Ys[i]) * Xs[i]
        )

    gradientValueTheta1 /= len(Xs)
    return gradientValueTheta1


def CalculateLoss():
    totalLoss = 0.0
    for i in range(len(Xs)):
        mse = (linearModel(theta1, Xs[i], theta0) - Ys[i]) ** 2
        totalLoss += mse

    totalLoss /= len(Xs)
    return totalLoss


# Training loop
for i in range(iterations):

    loss = CalculateLoss()
    gradientTheta0 = GetGradientTheta0()
    gradientTheta1 = GetGradientTheta1()

    print(
        f"Iteration: {i + 1}, "
        f"linear model: {theta1}x + {theta0}, "
        f"Loss: {loss}, "
        f"GradientTheta0: {gradientTheta0}, "
        f"GradientTheta1: {gradientTheta1}"
    )

    theta0 -= learningRate * gradientTheta0
    theta1 -= learningRate * gradientTheta1


print("Training done, pass input X value to get predictions:")
input_value = float(input())

result = linearModel(theta1, input_value, theta0)

print(f"Prediction for input {input_value}: {result}")
