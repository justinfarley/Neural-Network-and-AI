// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;

// var lines = File.ReadAllLines("data.csv");
// List<float> Xs = new List<float>();
// List<float> Ys = new List<float>();
// float learningRate = 0.01f;
// int iterations = 10000;

// Func<float, float, float, float> linearModel = (theta1, X, theta0) => theta1*X + theta0;

// float theta0 = 0;
// float theta1 = 0;


// foreach (var line in lines)
// {
//     var parts = line.Split(',');
//     Xs.Add(float.Parse(parts[0]));
//     Ys.Add(float.Parse(parts[1]));
// }

// for (int i = 0; i < iterations; i++)
// {
//     var loss = CalculateLoss();
//     var gradientTheta0 = GetGradientTheta0();
//     var gradientTheta1 = GetGradientTheta1();
//     Console.WriteLine($"Iteration: {i + 1}, linear model: {theta1}x + {theta0}, Loss: {loss}, GradientTheta0: {gradientTheta0}, GradientTheta1: {gradientTheta1}");
//     theta0 -= learningRate * gradientTheta0;
//     theta1 -= learningRate * gradientTheta1;
// }


// Console.WriteLine("Training done, pass input X value to get predictions: ");
// string input = Console.ReadLine();

// var result = linearModel(theta1, float.Parse(input), theta0);
// //result = result * stdDevY + meanY; //denormalize the output
// Console.WriteLine($"Prediction for input {input}: {result}");

// float GetGradientTheta0()
// {
//     float gradientValueTheta0 = 0;
//     for (int i = 0; i < Xs.Count; i++)
//     {
//         gradientValueTheta0 += (linearModel(theta1, Xs[i], theta0) - Ys[i]);
//     }
//     gradientValueTheta0 /= Xs.Count;

//     return gradientValueTheta0;
// }

// float GetGradientTheta1()
// {
//     float gradientValueTheta1 = 0;
//     for (int i = 0; i < Xs.Count; i++)
//     {
//         gradientValueTheta1 += (linearModel(theta1, Xs[i], theta0) - Ys[i]) * Xs[i];
//     }
//     gradientValueTheta1 /= Xs.Count;

//     return gradientValueTheta1;
// }

// float CalculateLoss()
// {
//     float totalLoss = 0;
//     for (int i = 0; i < Xs.Count; i++)
//     {
//         float MSE = MathF.Pow(linearModel(theta1, Xs[i], theta0) - Ys[i], 2);
//         totalLoss += MSE;
//     }
//     totalLoss /= Xs.Count;

//     return totalLoss;
// }