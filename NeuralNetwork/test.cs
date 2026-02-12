using Neural_Network_and_AI;

NeuralNetwork nn = new NeuralNetwork(0.0000001f, "data.csv", 2, 3);
nn.NetworkInit();
nn.Train(500);

