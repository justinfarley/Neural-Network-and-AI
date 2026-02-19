/// <summary>
/// Input node:
/// pretty straight forward, just holds the x value.
/// </summary>
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