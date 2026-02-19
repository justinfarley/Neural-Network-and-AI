using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Neural_Network_and_AI
{
    public class ModelExporter
    {
        private NeuralNetwork trainedNeuralNetwork;

        public ModelExporter(NeuralNetwork trainedNetwork)
        {
            trainedNeuralNetwork = trainedNetwork;
        }

        /// <summary>
        /// Exports the neural network model to a file
        /// Format: CSV-like with metadata header
        /// </summary>
        public void ExportModel(string path)
        {
            using (StreamWriter writer = new StreamWriter(path))
            {
                // Write metadata header
                writer.WriteLine($"LEARNING_RATE,{trainedNeuralNetwork.learningRate}");
                writer.WriteLine($"INPUT_SIZE,{trainedNeuralNetwork.inputSize}");
                writer.WriteLine($"OUTPUT_SIZE,{trainedNeuralNetwork.outputSize}");
                writer.WriteLine($"HIDDEN_LAYERS,{trainedNeuralNetwork.hiddenLayers}");
                writer.WriteLine($"HIDDEN_NODES_PER_LAYER,{trainedNeuralNetwork.hiddenNodesPerLayer}");
                writer.WriteLine("---"); // Separator
                
                // Export all weights
                writer.WriteLine("WEIGHTS");
                foreach (var weight in trainedNeuralNetwork.GetAllWeights())
                {
                    // Format: prevNodeIndex,nextNodeIndex,weightValue
                    int prevIndex = trainedNeuralNetwork.GetNodeIndex(weight.prev);
                    int nextIndex = trainedNeuralNetwork.GetNodeIndex(weight.next);
                    writer.WriteLine($"{prevIndex},{nextIndex},{weight.value}");
                }
                
                writer.WriteLine("---"); // Separator
                
                // Export all biases (for hidden and output nodes)
                writer.WriteLine("BIASES");
                
                // Hidden node biases
                foreach (var layer in trainedNeuralNetwork.GetHiddenLayers())
                {
                    foreach (var hiddenNode in layer.Value)
                    {
                        int nodeIndex = trainedNeuralNetwork.GetNodeIndex(hiddenNode);
                        writer.WriteLine($"{nodeIndex},{hiddenNode.bias}");
                    }
                }
                
                // Output node biases
                foreach (var outputNode in trainedNeuralNetwork.GetOutputNodes())
                {
                    int nodeIndex = trainedNeuralNetwork.GetNodeIndex(outputNode);
                    writer.WriteLine($"{nodeIndex},{outputNode.bias}");
                }
            }
            
            Console.WriteLine($"Model exported successfully to {path}");
        }

        /// <summary>
        /// Imports a neural network model from a file
        /// </summary>
        public static NeuralNetwork ImportModel(string path)
        {
            using (StreamReader reader = new StreamReader(path))
            {
                // Read metadata
                float learningRate = 0;
                int inputSize = 0;
                int outputSize = 0;
                int hiddenLayers = 0;
                int hiddenNodesPerLayer = 0;
                
                string line;
                while ((line = reader.ReadLine()) != null && line != "---")
                {
                    var parts = line.Split(',');
                    switch (parts[0])
                    {
                        case "LEARNING_RATE":
                            learningRate = float.Parse(parts[1]);
                            break;
                        case "INPUT_SIZE":
                            inputSize = int.Parse(parts[1]);
                            break;
                        case "OUTPUT_SIZE":
                            outputSize = int.Parse(parts[1]);
                            break;
                        case "HIDDEN_LAYERS":
                            hiddenLayers = int.Parse(parts[1]);
                            break;
                        case "HIDDEN_NODES_PER_LAYER":
                            hiddenNodesPerLayer = int.Parse(parts[1]);
                            break;
                    }
                }
                
                // Create network with same architecture
                var network = new NeuralNetwork(
                    learningRate,
                    inputSize,
                    outputSize,
                    hiddenLayers,
                    hiddenNodesPerLayer
                );
                network.NetworkInit();
                
                // Read weights section
                line = reader.ReadLine(); // Should be "WEIGHTS"
                var weights = network.GetAllWeights();
                int weightIndex = 0;
                
                while ((line = reader.ReadLine()) != null && line != "---")
                {
                    var parts = line.Split(',');
                    // We stored: prevNodeIndex,nextNodeIndex,weightValue
                    // But for loading, we just need to set weights in order
                    float weightValue = float.Parse(parts[2]);
                    weights[weightIndex].value = weightValue;
                    weightIndex++;
                }
                
                // Read biases section
                line = reader.ReadLine(); // Should be "BIASES"
                var nodeIndexToBias = new Dictionary<int, float>();
                
                while ((line = reader.ReadLine()) != null)
                {
                    var parts = line.Split(',');
                    int nodeIndex = int.Parse(parts[0]);
                    float biasValue = float.Parse(parts[1]);
                    nodeIndexToBias[nodeIndex] = biasValue;
                }
                
                // Apply biases
                foreach (var layer in network.GetHiddenLayers())
                {
                    foreach (var hiddenNode in layer.Value)
                    {
                        int nodeIndex = network.GetNodeIndex(hiddenNode);
                        if (nodeIndexToBias.ContainsKey(nodeIndex))
                        {
                            hiddenNode.bias = nodeIndexToBias[nodeIndex];
                        }
                    }
                }
                
                foreach (var outputNode in network.GetOutputNodes())
                {
                    int nodeIndex = network.GetNodeIndex(outputNode);
                    if (nodeIndexToBias.ContainsKey(nodeIndex))
                    {
                        outputNode.bias = nodeIndexToBias[nodeIndex];
                    }
                }
                
                Console.WriteLine($"Model imported successfully from {path}");
                return network;
            }
        }

    }
}