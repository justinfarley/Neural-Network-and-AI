using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GradientDescentApp
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