from Dataset import Dataset
import random
import os
from PIL import Image
from ImageProcessing import ImageProcessing
from NeuralNetwork import NeuralNetwork
from MathUtil import MathUtil

class FruitClassification(Dataset):

    def __init__(self):
        self.categories = [
            "Apple Braeburn", "Apple Granny Smith",
            "Apricot", "Banana", "Blueberry",
            "Cactus Fruit", "Cantaloupe",
            "Cherry", "Clementine", "Corn", "Cucumber Ripe",
            "Grape Blue", "Kiwi", "Lemon",
            "Limes", "Mango", "Onion White",
            "Orange", "Papaya", "Passion Fruit",
            "Peach", "Pear", "Pepper Green",
            "Pepper Red", "Pineapple", "Plum",
            "Pomegranate", "Potato Red", "Raspberry",
            "Strawberry", "Tomato", "Watermelon"
        ]

    def Testing(self, nn):

        test_path = "../fruitsTrainingData/test/test"
        testFiles = [os.path.join(test_path, f)
                     for f in os.listdir(test_path)
                     if f.endswith(".jpg")]

        for i in range(len(testFiles)):

            # Just predicting first 50 images
            if i == 50:
                break

            bitmap = Image.open(testFiles[i])
            floatArray = ImageProcessing.GetPixelColorsAsFloatArray(bitmap)

            print(f"File {testFiles[i]}")

            # Predict using Softmax
            results = nn.Predict(floatArray, NeuralNetwork.PredictionMethod.Softmax)

            bestClassification = max(results)
            percentagePrediction = bestClassification * 100
            idx = results.index(bestClassification)

            print(f"Prediction: {percentagePrediction}% chance to be a {self.categories[idx]}")


    def TrainingData(self, samples):

        # Map category -> one hot
        categoryToOneHot = {
            cat: MathUtil.GetOneHotEncodedArr(cat, self.categories)
            for cat in self.categories
        }

        # Load images
        imageFloatArrays = Dataset.LoadDataFromFileAll(
            "../fruitsTrainingData/train/train",
            ".jpg",
            32,
            32,
            self.categories,
            samples
        )

        allInputs = []
        allExpected = []

        # Shuffle
        random.shuffle(imageFloatArrays)

        # Take only requested sample count
        for category, imageArray in imageFloatArrays[:samples]:
            allInputs.append(imageArray)
            allExpected.append(list(categoryToOneHot[category]))

        print(f"Total images loaded: {len(imageFloatArrays)}, Total sampled: {samples}")

        return (allInputs, allExpected)
