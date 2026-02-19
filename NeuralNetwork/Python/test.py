from FruitClassification import FruitClassification
from CatsVsDogs import CatsVsDogs 
from NeuralNetwork import NeuralNetwork 
from ModelExporter import ModelExporter


# -------- SIMPLE LINEAR TEST --------
linearTestNN = NeuralNetwork(0.01, 5, 5, 2, 5)
linearTestNN.NetworkInit()
linearTestNN.TrainSingle(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [5.0, 1344.0, 714.0, 22.0, 100.0],
    5000
)
print(linearTestNN.Predict([1.0, 2.0, 3.0, 4.0, 5.0]))


fruitClassification = FruitClassification()
catsVsDogs = CatsVsDogs()


# ---------- CATS VS DOGS ------------
# To train a new model, uncomment and adjust values

# Compressed dimensions: 32x32 * 3 = 3072 inputs
# 2 outputs (cat vs dog)

# catsVsDogsNN = NeuralNetwork(0.003, 3072, 2, 2, 128)
# catsVsDogsNN.NetworkInit()
# allInputs, allOutputs = catsVsDogs.TrainingData(5000)
# catsVsDogsNN.Train(allInputs, allOutputs, 10, NeuralNetwork.PredictionMethod.Softmax)
# exporter = ModelExporter(catsVsDogsNN)
# exporter.ExportModel(f"CATSVSDOGSMODEL_{hash(catsVsDogsNN)}.csv")


# ---------- FRUIT CLASSIFICATION ------------
# To train a new model, uncomment and adjust values

# 3072 inputs (32x32x3)
# 32 output classes
# 3 hidden layers with 128 nodes each

# nn = NeuralNetwork(0.001, 3072, 32, 3, 128)
# nn.NetworkInit()
# allInputs, allExpected = fruitClassification.TrainingData(10000)
# nn.Train(allInputs, allExpected, 10, NeuralNetwork.PredictionMethod.Softmax)
# fruitClassification.Testing(nn)
# print("Trained!!!")
# exporter = ModelExporter(nn)
# exporter.ExportModel(f"FRUITCLASSIFICATIONMODEL_{hash(nn)}.csv")


# -------- LOAD PRETRAINED MODELS --------

#8+ hours of training on cats vs dogs CATSVSDOGSMODEL_1.csv
catsVsDogsNN = ModelExporter.ImportModel("CATSVSDOGSMODEL_1.csv")
catsVsDogs.Testing(catsVsDogsNN)

#8+ hours of training on fruit classification FRUITCLASSIFICATIONMODEL_1.csv
fruitClassificationNN = ModelExporter.ImportModel("FRUITCLASSIFICATIONMODEL_1.csv")
fruitClassification.Testing(fruitClassificationNN)
