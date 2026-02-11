using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GradientDescentApp
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
        }
    }
}