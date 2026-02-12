using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Neural_Network_and_AI
{
    public class Weight
    {
        public Node prev;
        public Node next;
        public float value;
    
        public Weight(Node prev, Node next, float value = 0.5f) 
        {
            this.prev = prev;
            this.next = next;
            this.value = value;
            
            prev.AddOutgoingWeight(this);
            next.AddIncomingWeight(this);
        }
    }
}