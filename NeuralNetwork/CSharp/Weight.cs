/// <summary>
/// The Weights!
/// Weights are the connections between the nodes. 
/// They have a value, and they also have a reference to the previous node and the next node
/// of which they are connected.
/// The value is their bias, which starts random and is adjusted during backpropagation.
/// </summary>
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