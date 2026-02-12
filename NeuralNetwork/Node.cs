/// <summary>
/// This is the base node class. 
/// It shows the basic properties that all 3 types of nodes (input, hidden, and output)
/// all have. 
/// Each node has a value, whether that be an x value, an activation value, or an output value.
/// They also have an error, and lists of incoming and outgoing weights. 
/// (this may not be optimal idk)
/// </summary>
namespace Neural_Network_and_AI
{
    public class Node
    {
        // base class for input and hidden nodes
        public float value;
        public float error;

        public List<Weight> outgoingWeights = new List<Weight>();
        public List<Weight> incomingWeights = new List<Weight>();

        public void AddOutgoingWeight(Weight weight)
        {
            outgoingWeights.Add(weight);
        }
        public void AddIncomingWeight(Weight weight)
        {
            incomingWeights.Add(weight);
        }
    }
}