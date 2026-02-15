using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using Neural_Network_and_AI;
using static Neural_Network_and_AI.MathUtil;

namespace Neural_Network_and_AI
{
    public class FruitClassification
    {

        private readonly string[] categories = 
        { "Apple Braeburn", "Apple Granny Smith", 
        "Apricot", "Banana", "Blueberry",
        "Cactus Fruit", "Cantaloupe", 
        "Cherry", "Clementine", "Corn", "Cucumber Ripe",
        "Grape Blue", "Kiwi", "Lemon",
        "Limes", "Mango", "Onion White",
        "Orange", "Papaya", "Passion Fruit",
        "Peach", "Pear", "Pepper Green",
        "Pepper Red", "Pineapple", "Plum",
        "Pomegranate", "Potato Red", "Raspberry",
        "Strawberry", "Tomato", "Watermelon" };
        public void Testing(NeuralNetwork nn)
        {
            var testFiles = Directory.GetFiles($"fruitsTrainingData/test/test", "*" + ".jpg");
            for(int i = 0; i < testFiles.Length; i++)
            {
                //Just predicting first 50 images for testing
                if(i == 50) break;
                Bitmap bitmap = new Bitmap(testFiles[i]);
                var floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap);
                Console.WriteLine($"File {testFiles[i]}");

                //Predict the test image with softmax and get results
                var results = nn.Predict(floatArray.ToList(), NeuralNetwork.PredictionMethod.Softmax);
                
                //Since we are using softmax, it converts the y values to percentages. The best percentage is the most accurate prediction
                float bestClassification = results.Max();
                float percentagePrediction = bestClassification * 100; //Convert to percentage
                int idx = results.IndexOf(bestClassification); //Corresponding category index

                //Print results!
                Console.WriteLine($"Prediction: {percentagePrediction}% chance to be a {categories[idx]}");
            }
        }

        public (List<List<float>>, List<List<float>>) TrainingData(int samples)
        {
            List<Tuple<string, List<float>>> imageFloatArrays = new List<Tuple<string, List<float>>>();
            
            //Store OneHot encoded outputs, example: [0f,0f,0f,0f,0f,1f,0f,0f,0f,...,0f] indicates 6th category output (Cactus Fruit)
            Dictionary<string, float[]> categoryToOneHot = categories.ToDictionary(cat => cat, cat => GetOneHotEncodedArr(cat, categories));

            //Get images from each category dir
            foreach(var category in categories)
            {
                var files = Directory.GetFiles($"fruitsTrainingData/train/train/{category}", "*" + ".jpg");
                Console.WriteLine($"Directory: {category}, Files found: {files.Length}");
                foreach(var imageFile in files)
                {
                    Bitmap bitmap = new Bitmap(imageFile);
                    var floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap);

                    //Store category and corresponding pixel float data for use later
                    imageFloatArrays.Add(new Tuple<string, List<float>>(category, floatArray.ToList()));
                }
            }

            Random random = new Random();
            List<List<float>> allInputs = new List<List<float>>();
            List<List<float>> allExpected = new List<List<float>>();

            //Shuffle the images so that we get random fruits and not all of one
            var shuffled = imageFloatArrays.OrderBy(x => random.Next()).ToList();

            //Only taking 1000 images due to testing / time constraints
            foreach (var (category, imageArray) in shuffled.Take(samples))
            {
                allInputs.Add(imageArray);
                allExpected.Add(categoryToOneHot[category].ToList());
            }

            Console.WriteLine($"Total images loaded: {imageFloatArrays.Count}");

            return (allInputs, allExpected);
        }

    }
}