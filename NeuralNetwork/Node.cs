using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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