using static Neural_Network_and_AI.MathUtil;

/// <summary>
/// Author: Justin Farley
/// 
/// This was fun to build
/// </summary>
namespace Neural_Network_and_AI
{
    public class NeuralNetwork
    {
        public float learningRate = 0.01f;
        private int hiddenLayers;
        private int hiddenNodesPerLayer;
        public List<float> Xs = new List<float>();
        public List<float> Ys = new List<float>();
        private List<InputNode> inputNodes = new List<InputNode>();
        private List<OutputNode> outputNodes = new List<OutputNode>();
        private Dictionary<int, List<HiddenNode>> hiddenNodes = new Dictionary<int, List<HiddenNode>>();
        private List<Weight> weights = new List<Weight>();


        public NeuralNetwork(float learningRate, List<float> Xs, List<float> Ys, int hiddenLayers = -1, int hiddenNodesPerLayer = -1)
        {
            this.learningRate = learningRate;
            this.Xs = Xs;
            this.Ys = Ys;
            this.hiddenLayers = hiddenLayers == -1 ? Xs.Count / 2 : hiddenLayers;
            this.hiddenNodesPerLayer = hiddenNodesPerLayer == -1 ? Xs.Count : hiddenNodesPerLayer;
        }

        public NeuralNetwork(float learningRate, string file, int hiddenLayers = -1, int hiddenNodesPerLayer = -1)
        {
            this.learningRate = learningRate;
            var lines = File.ReadAllLines(file);
            foreach (var line in lines)
            {
                var parts = line.Split(',');
                Xs.Add(float.Parse(parts[0]));
                Ys.Add(float.Parse(parts[1]));
            }
            this.hiddenLayers = hiddenLayers == -1 ? Xs.Count / 2 : hiddenLayers;
            this.hiddenNodesPerLayer = hiddenNodesPerLayer == -1 ? Xs.Count : hiddenNodesPerLayer;
        }

        public void NetworkInit()
        {
            Random random = new Random();        


            //make input nodes
            for(int i = 0; i < Xs.Count; i++)
            {
                InputNode input = new InputNode(Xs[i]);
                inputNodes.Add(input);
            }

            //Create first layer of hidden nodes to connect to input nodes
            hiddenNodes[0] = new List<HiddenNode>();    
            for(int j = 0; j < hiddenNodesPerLayer; j++)
            {
                HiddenNode hiddenNode = new HiddenNode((float)random.NextDouble());
                hiddenNodes[0].Add(hiddenNode);
            }

            //For every hidden node in the FIRST layer
            foreach(var hiddenNode in hiddenNodes[0])
            {
                foreach(var inputNode in inputNodes)
                {
                    Weight weight = new Weight(inputNode, hiddenNode, (float)random.NextDouble());
                    weights.Add(weight);
                }
            }

            //Starting from 1 to skip the initial layer we made already
            //Create these layers of hidden nodes
            for(int i = 1; i < hiddenLayers; i++)
            {
                hiddenNodes[i] = new List<HiddenNode>();
                for(int j = 0; j < hiddenNodesPerLayer; j++)
                {
                    HiddenNode hiddenNode = new HiddenNode((float)random.NextDouble());
                    hiddenNodes[i].Add(hiddenNode);
                }
            }

            foreach(KeyValuePair<int, List<HiddenNode>> kvp in hiddenNodes)
            {
                foreach(var hiddenNode in kvp.Value)
                {
                    if(kvp.Key == 0) continue; //skip the first layer since we already connected it to the input nodes
                    foreach(var prevHiddenNode in hiddenNodes[kvp.Key - 1])
                    {
                        Weight weight = new Weight(prevHiddenNode, hiddenNode, (float)random.NextDouble());
                        weights.Add(weight);
                    }
                }
            }

            //Now create output nodes
            for(int i = 0; i < Ys.Count; i++)
            {
                OutputNode outputNode = new OutputNode((float)random.NextDouble());
                outputNodes.Add(outputNode);
                foreach(var hiddenNode in hiddenNodes[hiddenLayers - 1])
                {
                    Weight weight = new Weight(hiddenNode, outputNode, (float)random.NextDouble());
                    weights.Add(weight);
                }
            }
        }

        public void Train(int iterations)
        {
            // implement training loop here
            for (int i = 0; i < iterations; i++)
            {
                // Forward pass
                foreach(var kvp in hiddenNodes)
                {
                    foreach(var hiddenNode in kvp.Value)
                    {
                        hiddenNode.Activation();
                    }
                }

                List<float> predictedValues = new List<float>();
                for(int j = 0; j < outputNodes.Count; j++)
                {
                    var outputNode = outputNodes[j];
                    outputNode.Prediction();
                    outputNode.error = GetOutputNodeError(outputNode.value, Ys[j]);
                    predictedValues.Add(outputNode.value);
                }

                Console.WriteLine($"Iteration: {i + 1} FORWARD PASS: Predictions: {string.Join(", ", predictedValues.Select((x, idx) => $"y{idx + 1}={x}"))}, loss = {CalculateLoss(predictedValues, Ys, Xs)}");


                //START BACKWARDS PASS
                Dictionary<Weight, float> weightGradients = new Dictionary<Weight, float>(); 
                Dictionary<Node, float> biasGradients = new Dictionary<Node, float>(); 
                foreach(var outputNode in outputNodes)
                {
                    foreach(var weight in outputNode.incomingWeights)
                    {
                        //SOLVE OUTPUT WEIGHT GRADIENTS
                        float gradient = GetOutputParameterGradient(outputNode.error, weight.prev.value);
                        weightGradients[weight] = gradient;
                    }
                    //SOLVE OUTPUT BIAS GRADIENTS
                    biasGradients[outputNode] = GetOutputParameterGradient(outputNode.error, 1f); //bias is always 1
                }

                //UPDATE HIDDEN ERRORS AND RELU DERIVATIVES
                Dictionary<HiddenNode, float> hiddenErrors = new Dictionary<HiddenNode, float>();
                Dictionary<HiddenNode, float> reluDerivatives = new Dictionary<HiddenNode, float>();
                foreach(var kvp in hiddenNodes)
                {
                    foreach(var hiddenNode in kvp.Value)
                    {
                        hiddenErrors[hiddenNode] = hiddenNode.HiddenNodeError();
                        reluDerivatives[hiddenNode] = hiddenNode.value > 0 ? 1 : 0;
                    }
                }
                // SOLVE HIDDEN LAYER(S) WEIGHT GRADIENTS AND INPUT LAYER WEIGHT GRADIENTS
                for(int j = hiddenLayers - 1; j >= 1; j--)
                {
                    Console.WriteLine($"Updating biases for hidden layer {j + 1}");
                    foreach(var hiddenNode in hiddenNodes[j])
                    {
                        foreach(var weight in hiddenNode.incomingWeights)
                        {
                            float gradient = GetMiddleLayerParameterGradient(hiddenErrors[hiddenNode], reluDerivatives[hiddenNode], weight.prev.value);
                            weightGradients[weight] = gradient;
                        }
                        biasGradients[hiddenNode] = GetMiddleLayerParameterGradient(hiddenErrors[hiddenNode], reluDerivatives[hiddenNode], 1f); //bias is always 1
                    }
                }

                //SOLVE INPUT LAYER WEIGHT GRADIENTS
                Console.WriteLine($"Updating biases for hidden layer 1");
                foreach(var hiddenNode in hiddenNodes[0])
                {
                    foreach(var weight in hiddenNode.incomingWeights)
                    {
                        float gradient = GetInputParameterGradient(hiddenErrors[hiddenNode], reluDerivatives[hiddenNode], weight.prev.value);
                        weightGradients[weight] = gradient;
                    }
                    biasGradients[hiddenNode] = GetInputParameterGradient(hiddenErrors[hiddenNode], reluDerivatives[hiddenNode], 1f); //bias is always 1
                }

                //UPDATE PARAMETERS!!
                foreach(var weight in weights)
                {
                    weight.value = UpdateParameter(weight.value, learningRate, weightGradients[weight]);
                }

                foreach(var kvp in hiddenNodes)
                {
                    foreach(var hiddenNode in kvp.Value)
                    {
                        hiddenNode.bias = UpdateParameter(hiddenNode.bias, learningRate, biasGradients[hiddenNode]);
                    }
                }   
            }
        }

    }
}