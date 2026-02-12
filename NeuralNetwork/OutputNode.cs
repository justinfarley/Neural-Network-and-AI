/// <summary>
/// Output Node:
/// Final node in the network, responsible for the prediction
/// of the Y value. Essentially the same as an activation function in a hidden node,
/// but we dont use the ReLU function, we take the raw output of the equation
/// as the predicted Y value.
/// </summary>
namespace Neural_Network_and_AI
{
    public class OutputNode : HiddenNode
    {
        public OutputNode(float bias) : base(bias)
        {
        }

        public void Prediction()
        {
            float prediction = bias;

            foreach(Weight weight in incomingWeights)
            {
                prediction += weight.value * weight.prev.value;
            }

            this.value = prediction;
        }
    }
}