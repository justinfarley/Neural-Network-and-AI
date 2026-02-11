using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using GradientDescentApp;

namespace GradientDescentApp
{
    public class HiddenNode : Node
    {
        public float bias;
        protected List<Weight> outgoingWeights;
        protected List<Weight> incomingWeights;
        

        public HiddenNode(float bias)
        {
            this.bias = bias;
            outgoingWeights = new List<Weight>();
            incomingWeights = new List<Weight>();
        }

        public void AddOutgoingWeight(Weight weight)
        {
            outgoingWeights.Add(weight);
        }
        public void AddIncomingWeight(Weight weight)
        {
            incomingWeights.Add(weight);
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