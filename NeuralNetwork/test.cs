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

NeuralNetwork nn = new NeuralNetwork(0.0000001f, "data.csv", 3, 15);
nn.NetworkInit();
nn.Train(50);

var testData1 = new List<float> { 1f, 2f, 3f, 4f, 5f };
var testData2 = new List<float> { 6f, 7f, 8f, 9f, 10f };

nn.Predict(testData1);  //should match data.csv
nn.Predict(testData2); //NNs prediction of 6,7,8,9,10
