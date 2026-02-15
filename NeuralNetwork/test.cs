using System.Drawing;
using System.Net.Http.Headers;
using Neural_Network_and_AI;

/// <summary>
/// Test File!! 
/// 
/// Currently creates a NN with learning rate 0.0000001f,
/// 3 hidden layers each with 15 nodes.
/// 
/// It trains for only 35 iterations and precisely predicts the Y values!
/// 
/// Adjusting the learning rate and # of hidden nodes can sometimes
/// cause the network to get stuck in a local minimum, or to exponentially explode with
/// too low of a learning rate.
/// 
/// </summary>

// Simple 1 pass NN with one list of floats as inputs
NeuralNetwork nn2 = new NeuralNetwork(0.01f, 5, 5, 2, 5);
nn2.NetworkInit();
nn2.Train(new List<float>(){1f,2f,3f,4f,5f}, new List<float>(){5f, 1344f, 714f, 22f, 100f}, 5000);
nn2.Predict(new List<float> { 1f,2f,3f,4f,5f }).ForEach(x => Console.Write(x + ", "));



//3072 pixels (compressed 32x32 img * 3 RGB values each = 3072), 32 fruit classes, 3 hidden layers with 128 nodes each.
//For fruit classification
//NeuralNetwork nn = new NeuralNetwork(0.001f, 3072, 32, 3, 128);
FruitClassification fruitClassification = new FruitClassification();

//Train neural network based on all of the inputs and expected data for 50 iterations using Softmax for classification
//nn.NetworkInit();
//(var allInputs, var allExpected) = fruitClassification.TrainingData(100);
//nn.Train(allInputs, allExpected, 10, NeuralNetwork.PredictionMethod.Softmax); //Softmax for classification
//fruitClassification.Testing(nn);
//Console.WriteLine($"Trained!!!");
//ModelExporter exporter = new ModelExporter(nn);
//exporter.ExportModel($"FRUITCLASSIFICATIONMODEL_{nn.GetHashCode()}.csv");


NeuralNetwork loadedModel = ModelExporter.ImportModel("FRUITCLASSIFICATIONMODEL_1.csv");
fruitClassification.Testing(loadedModel);





