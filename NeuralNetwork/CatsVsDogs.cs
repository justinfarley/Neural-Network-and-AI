using System.Drawing;
using static Neural_Network_and_AI.MathUtil;
namespace Neural_Network_and_AI
{
    public class CatsVsDogs : Dataset
    {
        private readonly string[] categories =
        [
            "Cat",
            "Dog"
        ];
        private int compressedWidth = 32;
        private int compressedHeight = 32;
        public void Testing(NeuralNetwork nn)
        {
            Console.WriteLine("Input path to a photo of a cat or dog:");
            var inputPath = Console.ReadLine();
            Bitmap bitmap = new Bitmap(inputPath);
            var floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap, compressedWidth, compressedHeight);
            var results = nn.Predict(floatArray.ToList(), NeuralNetwork.PredictionMethod.Softmax);
            Console.WriteLine($"Probability of photo being a Cat: {results[0] * 100}%");
            Console.WriteLine($"Probability of photo being a Dog: {results[1] * 100}%");
        }

        public (List<List<float>>, List<List<float>>) TrainingData(int samples = -1)
        {
            List<Tuple<string, List<float>>> imageFloatArrays;
            Dictionary<string, float[]> categoryToOneHot = categories.ToDictionary(cat => cat, cat => GetOneHotEncodedArr(cat, categories));

            imageFloatArrays = Dataset.LoadDataFromFileAll("PetImages", ".jpg", ImageProcessing.GetPixelColorsAsFloatArray, compressedWidth, compressedHeight, categories, samples);

            Random random = new Random(15235);
            List<List<float>> allInputs = new List<List<float>>();
            List<List<float>> allExpected = new List<List<float>>();

            //Shuffle the images so that we get random and not all of one category
            var shuffled = imageFloatArrays.OrderBy(x => random.Next()).ToList();

            foreach (var (category, imageArray) in shuffled.Take(samples))
            {
                allInputs.Add(imageArray);
                allExpected.Add(categoryToOneHot[category].ToList());
            }

            Console.WriteLine($"Total images loaded: {imageFloatArrays.Count}, Total sampled: {samples}");

            return (allInputs, allExpected);
        }
    }
}