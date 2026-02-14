using System.Drawing;
using Neural_Network_and_AI;

/// <summary>
/// Test File!! 
/// 
/// Currently creates a NN with learning rate 0.0000001f,
/// 3 hidden layers each with 15 nodes.
/// 
/// It trains for only 35 iterations and precisely predicts the Y values!
/// 
/// Adjusting the learning rate and # of hidden nodes can sometimes
/// cause the network to get stuck in a local minimum, or to exponentially explode with
/// too low of a learning rate.
/// 
/// </summary>

List<Tuple<string, List<float>>> imageFloatArrays = new List<Tuple<string, List<float>>>();
string imgExtension = ".jpg";
var imageFiles = new List<string>();
List<float> outputLabels = new List<float>();

// Debug: Check current working directory
Console.WriteLine($"Current working directory: {Directory.GetCurrentDirectory()}");

NeuralNetwork nn2 = new NeuralNetwork(0.01f, 5, 5, 2, 5);
nn2.NetworkInit();
nn2.Train(new List<float>(){1f,2f,3f,4f,5f}, new List<float>(){5f, 1344f, 714f, 22f, 100f}, 5000);
nn2.Predict(new List<float> { 1f,2f,3f,4f,5f }).ForEach(x => Console.Write(x + ", "));



// Navigate up to project root and to fruitsTrainingData
string basePath = Path.Combine(Directory.GetCurrentDirectory(), "../../../fruitsTrainingData/train/train");
string[] categories = 
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

float[] GetOneHotEncodedArr(string category, string[] categories)
{
    float[] vector = new float[categories.Length];
    for (int i = 0; i < categories.Length; i++)
    {
        vector[i] = categories[i] == category ? 1.0f : 0.0f;
    }
    return vector;
}
Dictionary<string, float[]> categoryToOneHot = categories.ToDictionary(cat => cat, cat => GetOneHotEncodedArr(cat, categories));

foreach(var imageDirectory in categories)
{
    var files = Directory.GetFiles($"fruitsTrainingData/train/train/{imageDirectory}", "*" + imgExtension);
    Console.WriteLine($"Directory: {imageDirectory}, Files found: {files.Length}");
    foreach(var imageFile in files)
    {
        Bitmap bitmap = new Bitmap(imageFile);
        var floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap);
        imageFloatArrays.Add(new Tuple<string, List<float>>(imageDirectory, floatArray.ToList()));
    }
    outputLabels.Add(Array.IndexOf(categories, imageDirectory));
}

Console.WriteLine($"Total images loaded: {imageFloatArrays.Count}, Labels: {outputLabels.Count}");

if (imageFloatArrays.Count == 0)
{
    Console.WriteLine("ERROR: No images loaded! Check your directory paths.");
    return;
}

//3072 pixels (compressed 32x32 img), 32 fruit classes, 3 hidden layers with 128 nodes each.
NeuralNetwork nn = new NeuralNetwork(0.001f, 3072, 32, 3, 128);
nn.NetworkInit();

Random random = new Random();
List<List<float>> allInputs = new List<List<float>>();
List<List<float>> allExpected = new List<List<float>>();
var shuffled = imageFloatArrays.OrderBy(x => random.Next()).ToList();
foreach (var (category, imageArray) in shuffled.Take(1000))
{
    allInputs.Add(imageArray);
    allExpected.Add(categoryToOneHot[category].ToList());
}
nn.Train(allInputs, allExpected, 50, NeuralNetwork.PredictionMethod.Softmax); // Train 1 iteration per image
Console.WriteLine($"Trained!!!");

var testFiles = Directory.GetFiles($"fruitsTrainingData/test/test", "*" + imgExtension);
for(int i = 0; i < testFiles.Length; i++)
{
    if(i == 50) break;
    Bitmap bitmap = new Bitmap(testFiles[i]);
    var floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap);
    Console.WriteLine($"File {testFiles[i]}");
    var results = nn.Predict(floatArray.ToList(), NeuralNetwork.PredictionMethod.Softmax);
    float bestClassification = results.Max();
    float percentagePrediction = bestClassification * 100;
    int idx = results.IndexOf(bestClassification);

    Console.WriteLine($"Prediction: {percentagePrediction}% chance to be a {categories[idx]}");

}

//TODO: implement exporting models and loading them so that you can train once and use a model
