using System.ComponentModel;
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
        public enum PredictionMethod{
            Linear,
            Softmax
        }
        public float learningRate = 0.01f;
        private int hiddenLayers;
        private int hiddenNodesPerLayer;
        private int inputSize;
        private int outputSize;
        private List<InputNode> inputNodes = new List<InputNode>();
        private List<OutputNode> outputNodes = new List<OutputNode>();
        private Dictionary<int, List<HiddenNode>> hiddenNodes = new Dictionary<int, List<HiddenNode>>();
        private List<Weight> weights = new List<Weight>();


        public NeuralNetwork(float learningRate, int inputSize, int outputSize, int hiddenLayers, int hiddenNodesPerLayer)
        {
            this.learningRate = learningRate;
            this.hiddenLayers = hiddenLayers;
            this.hiddenNodesPerLayer = hiddenNodesPerLayer;
            this.inputSize = inputSize;
            this.outputSize = outputSize;
        }

        private float GetWeightValue(Random random, int prevLayerCount)
        {
            float fanIn = prevLayerCount; // number of incoming connections
            float scale = MathF.Sqrt(2f / fanIn);
            float weightValue = ((float)random.NextDouble() * 2f - 1f) * scale;
            return weightValue;
        }

        public void NetworkInit()
        {
            Random random = new Random();        


            //make input nodes
            for(int i = 0; i < inputSize; i++)
            {
                InputNode input = new InputNode(0);
                inputNodes.Add(input);
            }

            //Create first layer of hidden nodes to connect to input nodes
            hiddenNodes[0] = new List<HiddenNode>();    
            for(int j = 0; j < hiddenNodesPerLayer; j++)
            {
                HiddenNode hiddenNode = new HiddenNode(GetWeightValue(random, inputNodes.Count));
                hiddenNodes[0].Add(hiddenNode);
            }

            //For every hidden node in the FIRST layer
            foreach(var hiddenNode in hiddenNodes[0])
            {
                foreach(var inputNode in inputNodes)
                {
                    Weight weight = new Weight(inputNode, hiddenNode, GetWeightValue(random, inputNodes.Count));
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
                    HiddenNode hiddenNode = new HiddenNode(GetWeightValue(random, hiddenNodesPerLayer));
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
                        Weight weight = new Weight(prevHiddenNode, hiddenNode, GetWeightValue(random, hiddenNodesPerLayer));
                        weights.Add(weight);
                    }
                }
            }

            //Now create output nodes
            for(int i = 0; i < outputSize; i++)
            {
                OutputNode outputNode = new OutputNode((float)random.NextDouble());
                outputNodes.Add(outputNode);
                foreach(var hiddenNode in hiddenNodes[hiddenLayers - 1])
                {
                    Weight weight = new Weight(hiddenNode, outputNode, GetWeightValue(random, hiddenNodesPerLayer));
                    weights.Add(weight);
                }
            }
        }

        private List<float> ForwardPass(List<float> input, PredictionMethod method = PredictionMethod.Linear, List<float> expected = null)
        {

            for(int i = 0; i < inputNodes.Count; i++)
            {
                inputNodes[i].value = input[i];
            }

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
                if(expected != null && method == PredictionMethod.Linear)
                    outputNode.error = GetOutputNodeError(outputNode.value, expected[j]);
                predictedValues.Add(outputNode.value);
            }
            if(method != PredictionMethod.Softmax)
                Console.WriteLine($"Loss (MSE): {MathUtil.MSELoss(expected ?? new List<float>(), predictedValues)}");

            if(method == PredictionMethod.Softmax)
            {
                predictedValues = new List<float>();
                var rawOutputs = outputNodes.Select(x => x.value).ToArray();

                float max = rawOutputs.Max();
                float sums = rawOutputs.Sum(x => MathF.Exp(x - max));

                for (int i = 0; i < outputNodes.Count; i++)
                {
                    float softmaxVal = MathF.Exp(rawOutputs[i] - max) / sums;
                    outputNodes[i].value = softmaxVal;
                    predictedValues.Add(softmaxVal);

                    if(expected != null)
                        outputNodes[i].error = softmaxVal - expected[i];
                }
            }
            return predictedValues;
        }

        public void BackwardsPass()
        {
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
                for (int j = hiddenLayers - 1; j >= 0; j--)
                {
                    foreach (var hiddenNode in hiddenNodes[j])
                    {
                        float error = 0;
                        foreach (var outgoingWeight in hiddenNode.outgoingWeights)
                        {
                            error += outgoingWeight.next.error * outgoingWeight.value;
                        }
                        hiddenNode.error = error * (hiddenNode.value > 0 ? 1.0f : 0.0f); 
                    }
                }
                // SOLVE HIDDEN LAYER(S) WEIGHT GRADIENTS AND INPUT LAYER WEIGHT GRADIENTS
                for(int j = hiddenLayers - 1; j >= 1; j--)
                {
                    foreach(var hiddenNode in hiddenNodes[j])
                    {
                        foreach(var weight in hiddenNode.incomingWeights)
                        {
                            float gradient = hiddenNode.error * weight.prev.value;;
                            weightGradients[weight] = gradient;
                        }
                        biasGradients[hiddenNode] = hiddenNode.error; //bias is always 1
                    }
                }

                //SOLVE INPUT LAYER WEIGHT GRADIENTS
                foreach(var hiddenNode in hiddenNodes[0])
                {
                    foreach(var weight in hiddenNode.incomingWeights)
                    {
                        float gradient = hiddenNode.error * weight.prev.value;
                        weightGradients[weight] = gradient;
                    }
                    biasGradients[hiddenNode] = hiddenNode.error; //bias is always 1
                }

                //UPDATE PARAMETERS!!
                foreach(var weight in weights)
                {
                    weight.value = UpdateParameter(weight.value, learningRate, weightGradients[weight]);
                }

                //Console.WriteLine($"First weight: {inputNodes[0].outgoingWeights[0].value}");

                foreach(var kvp in hiddenNodes)
                {
                    foreach(var hiddenNode in kvp.Value)
                    {
                        hiddenNode.bias = UpdateParameter(hiddenNode.bias, learningRate, biasGradients[hiddenNode]);
                    }
                }   
                
                // Apply output node bias updates (if computed)
                foreach (var outputNode in outputNodes)
                {
                    outputNode.bias = UpdateParameter(outputNode.bias, learningRate, biasGradients[outputNode]);
                }
                weightGradients.Clear();
                biasGradients.Clear();
        }

        public List<float> Predict(List<float> input, PredictionMethod method = PredictionMethod.Linear)
        {
            List<float> results = ForwardPass(input, method);
            return results;
        }

        public void Train(List<List<float>> inputs, List<List<float>> expected, int iterations, PredictionMethod method = PredictionMethod.Linear)
        {
            for(int j = 0; j < iterations; j++)
            {
                float totalLoss = 0;
                for(int i = 0; i < inputs.Count; i++)
                {
                    var preds = ForwardPass(inputs[i], method, expected[i]);

                    BackwardsPass();              
                    totalLoss += CrossEntropyLoss(preds, expected[i]);
                }
                float avgLoss = totalLoss / inputs.Count;
                Console.WriteLine($"Epoch {j + 1}/{iterations} - Avg Loss: {avgLoss}");
            }
        }

        public void Train(List<float> inputs, List<float> expected, int iterations, PredictionMethod method = PredictionMethod.Linear)
        {
            for (int i = 0; i < iterations; i++)
            {
                // Forward pass
                ForwardPass(inputs, method, expected);

                BackwardsPass();
            }
        }
    }
}