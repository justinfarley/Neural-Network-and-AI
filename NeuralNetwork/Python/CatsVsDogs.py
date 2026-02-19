from Dataset import Dataset
import random
from PIL import Image
from ImageProcessing import ImageProcessing
from NeuralNetwork import NeuralNetwork
from MathUtil import MathUtil
class CatsVsDogs(Dataset):

    def __init__(self):
        self.categories = [
            "Cat",
            "Dog"
        ]

        self.compressedWidth = 32
        self.compressedHeight = 32

    def Testing(self, nn):

        print("Input path to a photo of a cat or dog:")
        inputPath = input()

        bitmap = Image.open(inputPath)

        floatArray = ImageProcessing.GetPixelColorsAsFloatArray(
            bitmap,
            self.compressedWidth,
            self.compressedHeight
        )

        results = nn.Predict(
            floatArray,
            NeuralNetwork.PredictionMethod.Softmax
        )

        print(f"Probability of photo being a Cat: {results[0] * 100}%")
        print(f"Probability of photo being a Dog: {results[1] * 100}%")

    def TrainingData(self, samples=-1):

        categoryToOneHot = {
            cat: MathUtil.GetOneHotEncodedArr(cat, self.categories)
            for cat in self.categories
        }

        imageFloatArrays = Dataset.LoadDataFromFileAll(
            "../PetImages",
            ".jpg",
            self.compressedWidth,
            self.compressedHeight,
            self.categories,
            samples
        )

        allInputs = []
        allExpected = []

        random.seed(15235)
        random.shuffle(imageFloatArrays)

        # If samples == -1, take all
        if samples == -1:
            selected = imageFloatArrays
        else:
            selected = imageFloatArrays[:samples]

        for category, imageArray in selected:
            allInputs.append(imageArray)
            allExpected.append(list(categoryToOneHot[category]))

        print(f"Total images loaded: {len(imageFloatArrays)}, Total sampled: {samples}")

        return (allInputs, allExpected)
