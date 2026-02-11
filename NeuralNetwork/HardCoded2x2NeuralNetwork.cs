using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GradientDescentApp;
using static GradientDescentApp.MathUtil;

var lines = File.ReadAllLines("data.csv");
List<float> Xs = new List<float>();
List<float> Ys = new List<float>();
float learningRate = 0.01f;
int iterations = 50;

foreach (var line in lines)
{
    var parts = line.Split(',');
    Xs.Add(float.Parse(parts[0]));
    Ys.Add(float.Parse(parts[1]));
}

InputNode x1Node = new InputNode(Xs[0]); // Initialize with first X value
InputNode x2Node = new InputNode(Xs[1]);

HiddenNode h1 = new HiddenNode(0.1f);
HiddenNode h2 = new HiddenNode(0.2f);

OutputNode y1 = new OutputNode(0.3f);
OutputNode y2 = new OutputNode(0.5f);

Weight w11 = new Weight(x1Node, h1, 0.5f);
Weight w12 = new Weight(x1Node, h2, 0.3f);
Weight w21 = new Weight(x2Node, h1, 0.4f);
Weight w22 = new Weight(x2Node, h2, 0.6f);

h1.AddIncomingWeight(w11);
h1.AddIncomingWeight(w12);
h2.AddIncomingWeight(w21);
h2.AddIncomingWeight(w22);

Weight v11 = new Weight(h1, y1, 0.2f);
Weight v12 = new Weight(h1, y2, 0.5f);
Weight v21 = new Weight(h2, y1, 0.3f);
Weight v22 = new Weight(h2, y2, 0.1f);

h1.AddOutgoingWeight(v11);
h1.AddOutgoingWeight(v12);
h2.AddOutgoingWeight(v21);
h2.AddOutgoingWeight(v22);

y1.AddIncomingWeight(v11);
y1.AddIncomingWeight(v21);
y2.AddIncomingWeight(v12);
y2.AddOutgoingWeight(v22);

for (int i = 0; i < iterations; i++)
{
    // Forward pass
    h1.Activation();
    h2.Activation();
    y1.Prediction();
    y2.Prediction();

    y1.error = GetOutputNodeError(y1.value, Ys[0]);
    y2.error = GetOutputNodeError(y2.value, Ys[1]);

    List<float> predictedValues = new List<float> { y1.value, y2.value };


    // Calculate loss and gradients, then update weights (not implemented here)
    Console.WriteLine($"Iteration: {i + 1} FORWARD PASS: Predictions: y1={y1.value}, y2={y2.value}, loss = {CalculateLoss(predictedValues, Ys, Xs)}");

    // Update all weights and biases, backpropagation here
    float v11Gradient = GetOutputParameterGradient(y1.error, h1.value);
    float v12Gradient = GetOutputParameterGradient(y2.error, h1.value);
    float v21Gradient = GetOutputParameterGradient(y1.error, h2.value);
    float v22Gradient = GetOutputParameterGradient(y2.error, h2.value);
    

    List<float> hiddenErrors = new List<float> { h1.HiddenNodeError(), h2.HiddenNodeError() };
    List<float> reluDerivatives = new List<float> { h1.value > 0 ? 1 : 0, h2.value > 0 ? 1 : 0 };

    float w11Gradient = GetInputParameterGradient(hiddenErrors[0], reluDerivatives[0], Xs[0]);
    float w21Gradient = GetInputParameterGradient(hiddenErrors[0], reluDerivatives[0], Xs[1]);
    float w12Gradient = GetInputParameterGradient(hiddenErrors[1], reluDerivatives[1], Xs[0]);
    float w22Gradient = GetInputParameterGradient(hiddenErrors[1], reluDerivatives[1], Xs[1]);
    
    float biasO1Gradient = GetInputParameterGradient(hiddenErrors[0], reluDerivatives[0], 1);
    float biasO2Gradient = GetInputParameterGradient(hiddenErrors[1], reluDerivatives[1], 1);

    //parameter update
    v11.value = UpdateParameter(v11.value, learningRate, v11Gradient);
    v12.value = UpdateParameter(v12.value, learningRate, v12Gradient);
    v21.value = UpdateParameter(v21.value, learningRate, v21Gradient);
    v22.value = UpdateParameter(v22.value, learningRate, v22Gradient);

    w11.value = UpdateParameter(w11.value, learningRate, w11Gradient);
    w21.value = UpdateParameter(w21.value, learningRate, w21Gradient);
    w12.value = UpdateParameter(w12.value, learningRate, w12Gradient);
    w22.value = UpdateParameter(w22.value, learningRate, w22Gradient);

    h1.bias = UpdateParameter(h1.bias, learningRate, biasO1Gradient);
    h2.bias = UpdateParameter(h2.bias, learningRate, biasO2Gradient);

}