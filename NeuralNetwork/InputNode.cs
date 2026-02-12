using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Neural_Network_and_AI
{
    public class InputNode : Node
    {
        public InputNode(float xValue)
        {
            this.value = xValue;
        }
    }
}