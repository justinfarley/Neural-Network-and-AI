/// <summary>
/// Just a Util class. Some methods repeat but the naming was helping me 
/// keep track of the backpropagation process, so I left them as is.
/// </summary>
namespace Neural_Network_and_AI
{
    public static class MathUtil
    {
        public static float[] GetOneHotEncodedArr(string category, string[] categories)
        {
            float[] vector = new float[categories.Length];
            for (int i = 0; i < categories.Length; i++)
            {
                vector[i] = categories[i] == category ? 1.0f : 0.0f;
            }
            return vector;
        }
        
        public static float MSELoss(List<float> predictedValues, List<float> actualValues)
        {
            float totalLoss = 0;
            for (int i = 0; i < predictedValues.Count; i++)
            {
                float MSE = MathF.Pow(predictedValues[i] - actualValues[i], 2);
                totalLoss += MSE;
            }
            totalLoss /= predictedValues.Count;

            return totalLoss;
        }

        public static float CrossEntropyLoss(List<float> predicted, List<float> actual)
        {
            if(actual.Count <= 0) return -1;

            float loss = 0;
            for (int i = 0; i < predicted.Count; i++)
            {
                float p = Math.Max(predicted[i], 1e-15f);
                loss -= actual[i] * MathF.Log(p);
            }
            return loss;
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