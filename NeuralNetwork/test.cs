using Neural_Network_and_AI;

NeuralNetwork nn = new NeuralNetwork(0.0000001f, "data.csv", 3, 15);
nn.NetworkInit();
nn.Train(35);

