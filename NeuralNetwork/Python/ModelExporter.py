from NeuralNetwork import NeuralNetwork
class ModelExporter:

    def __init__(self, trainedNetwork):
        self.trainedNeuralNetwork = trainedNetwork

    """
    Exports the neural network model to a file
    Format: CSV-like with metadata header
    """
    def ExportModel(self, path):

        with open(path, "w") as writer:

            # Metadata
            writer.write(f"LEARNING_RATE,{self.trainedNeuralNetwork.learningRate}\n")
            writer.write(f"INPUT_SIZE,{self.trainedNeuralNetwork.inputSize}\n")
            writer.write(f"OUTPUT_SIZE,{self.trainedNeuralNetwork.outputSize}\n")
            writer.write(f"HIDDEN_LAYERS,{self.trainedNeuralNetwork.hiddenLayers}\n")
            writer.write(f"HIDDEN_NODES_PER_LAYER,{self.trainedNeuralNetwork.hiddenNodesPerLayer}\n")
            writer.write("---\n")

            # Weights
            writer.write("WEIGHTS\n")

            for weight in self.trainedNeuralNetwork.GetAllWeights():

                prevIndex = self.trainedNeuralNetwork.GetNodeIndex(weight.prev)
                nextIndex = self.trainedNeuralNetwork.GetNodeIndex(weight.next)

                writer.write(f"{prevIndex},{nextIndex},{weight.value}\n")

            writer.write("---\n")

            # Biases
            writer.write("BIASES\n")

            # Hidden node biases
            for layer in self.trainedNeuralNetwork.GetHiddenLayers().values():
                for hiddenNode in layer:
                    nodeIndex = self.trainedNeuralNetwork.GetNodeIndex(hiddenNode)
                    writer.write(f"{nodeIndex},{hiddenNode.bias}\n")

            # Output node biases
            for outputNode in self.trainedNeuralNetwork.GetOutputNodes():
                nodeIndex = self.trainedNeuralNetwork.GetNodeIndex(outputNode)
                writer.write(f"{nodeIndex},{outputNode.bias}\n")

        print(f"Model exported successfully to {path}")
    @staticmethod
    def ImportModel(path):

        with open(path, "r") as reader:

            # ---- Read Metadata ----
            learningRate = 0.0
            inputSize = 0
            outputSize = 0
            hiddenLayers = 0
            hiddenNodesPerLayer = 0

            while True:
                line = reader.readline().strip()
                if line == "---":
                    break

                parts = line.split(",")

                if parts[0] == "LEARNING_RATE":
                    learningRate = float(parts[1])
                elif parts[0] == "INPUT_SIZE":
                    inputSize = int(parts[1])
                elif parts[0] == "OUTPUT_SIZE":
                    outputSize = int(parts[1])
                elif parts[0] == "HIDDEN_LAYERS":
                    hiddenLayers = int(parts[1])
                elif parts[0] == "HIDDEN_NODES_PER_LAYER":
                    hiddenNodesPerLayer = int(parts[1])

            # ---- Recreate Network ----
            network = NeuralNetwork(
                learningRate,
                inputSize,
                outputSize,
                hiddenLayers,
                hiddenNodesPerLayer
            )

            network.NetworkInit()

            # ---- Read Weights ----
            reader.readline()  # Skip "WEIGHTS"

            weights = network.GetAllWeights()
            weightIndex = 0

            while True:
                line = reader.readline().strip()
                if line == "---":
                    break

                parts = line.split(",")
                weightValue = float(parts[2])

                weights[weightIndex].value = weightValue
                weightIndex += 1

            # ---- Read Biases ----
            reader.readline()  # Skip "BIASES"

            nodeIndexToBias = {}

            for line in reader:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                nodeIndex = int(parts[0])
                biasValue = float(parts[1])
                nodeIndexToBias[nodeIndex] = biasValue

            # Apply biases to hidden nodes
            for layer in network.GetHiddenLayers().values():
                for hiddenNode in layer:
                    nodeIndex = network.GetNodeIndex(hiddenNode)
                    if nodeIndex in nodeIndexToBias:
                        hiddenNode.bias = nodeIndexToBias[nodeIndex]

            # Apply biases to output nodes
            for outputNode in network.GetOutputNodes():
                nodeIndex = network.GetNodeIndex(outputNode)
                if nodeIndex in nodeIndexToBias:
                    outputNode.bias = nodeIndexToBias[nodeIndex]

            print(f"Model imported successfully from {path}")

            return network