/// <summary>
/// The Hidden Node. Used in the hidden layers of the neural network
/// between the input and output layers. Each node has an activation function which is 
/// w1x1 + w2x2 + ... + b, where w is the weight of the connection, 
/// x is the x value, and b is the bias. 
/// The equation is then passed through a ReLU function to see if the neuron "fires" or not.
/// </summary>
namespace Neural_Network_and_AI
{
    public class HiddenNode : Node
    {
        public float bias;
        public HiddenNode(float bias)
        {
            this.bias = bias;
        }



        //z1 = w1x1 + w2x2 + b
        public void Activation()
        {
            float activation = bias;

            foreach(Weight weight in incomingWeights)
            {
                activation += weight.value * weight.prev.value;
            }

            activation = MathF.Max(0, activation); // ReLU activation function

            this.value = activation;
        }

        public float HiddenNodeError()
        {
            float error = 0;
            foreach(Weight weight in outgoingWeights)
            {
                error += weight.value * weight.next.error; 
            }
            this.error = error;
            return error;
        }
    }
}