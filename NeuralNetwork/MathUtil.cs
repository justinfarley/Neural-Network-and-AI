using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GradientDescentApp
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

        public static float GetOutputParameterGradient(float error, float activation)
        {
            return error * activation;
        }

        public static float GetInputParameterGradient(float hiddenError, float reluDerivative, float xValue)
        {
            return hiddenError * reluDerivative * xValue;
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