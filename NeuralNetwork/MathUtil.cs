/// <summary>
/// Just a Util class. Some methods repeat but the naming was helping me 
/// keep track of the backpropagation process, so I left them as is.
/// </summary>
namespace Neural_Network_and_AI
{
    public static class MathUtil
    {
        public static float CalculateLoss(List<float> predictedValues, List<float> actualValues, List<float> xValues)
        {
            float totalLoss = 0;
            for (int i = 0; i < xValues.Count; i++)
            {
                float MSE = MathF.Pow(predictedValues[i] - actualValues[i], 2);
                totalLoss += MSE;
            }
            totalLoss /= xValues.Count;

            return totalLoss;
        }

        public static float GetOutputParameterGradient(float error, float activationPrevLayer)
        {
            return error * activationPrevLayer;
        }

        public static float GetInputParameterGradient(float hiddenError, float reluDerivative, float xValue)
        {
            return hiddenError * reluDerivative * xValue;
        }
        
        public static float GetMiddleLayerParameterGradient(float hiddenError, float reluDerivative, float activationPrevLayer)
        {
            return hiddenError * reluDerivative * activationPrevLayer;
        }

        public static float UpdateParameter(float oldParameter, float learningRate, float gradient)
        {
            return oldParameter - learningRate * gradient;
        }

        public static float GetOutputNodeError(float predictedValue, float actualValue)
        {
            return predictedValue - actualValue;
        }
    }
}