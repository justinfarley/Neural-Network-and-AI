from ImageProcessing import ImageProcessing
from PIL import Image
import os
class Dataset:

    # The Neural Network to test on
    def Testing(self, nn):
        raise NotImplementedError("Testing method must be implemented.")

    """
    Returns:
        (list_of_inputs, list_of_expected_outputs)
    """
    def TrainingData(self, samples=-1):
        raise NotImplementedError("TrainingData method must be implemented.")

    @staticmethod
    def LoadDataFromFileAll(path,
                            extension,
                            compressionX,
                            compressionY,
                            categories,
                            samples):

        ret = []
        samplesPerCategory = -1

        if samples > 0:
            samplesPerCategory = samples // len(categories)

        for category in categories:

            folder_path = f"{path}/{category}"
            files = [f for f in os.listdir(folder_path)
                     if f.endswith(extension)]

            print(f"Directory: {folder_path}, Files found: {len(files)}")

            count = 0
            for imageFile in files:
                count += 1

                if samples > 0 and count > samplesPerCategory:
                    print(f"Directory: {folder_path}, "
                          f"Files loaded in sample: {len(ret)}/{samples}")
                    break

                try:
                    img_path = os.path.join(folder_path, imageFile)
                    bitmap = Image.open(img_path)

                    floatArray = ImageProcessing.GetPixelColorsAsFloatArray(
                        bitmap,
                        compressionX,
                        compressionY
                    )

                    ret.append((category, list(floatArray)))

                except Exception as e:
                    print(str(e) + f" IGNORING THIS FILE {imageFile}")

        return ret


    @staticmethod
    def LoadDataFromFile(path,
                         category,
                         extension,
                         compressionX,
                         compressionY):

        ret = []

        folder_path = f"{path}/{category}"
        files = [f for f in os.listdir(folder_path)
                 if f.endswith(extension)]

        print(f"Directory: {category}, Files found: {len(files)}")

        for imageFile in files:
            try:
                img_path = os.path.join(folder_path, imageFile)
                bitmap = Image.open(img_path)

                floatArray = ImageProcessing.GetPixelColorsAsFloatArray(
                    bitmap,
                    compressionX,
                    compressionY
                )

                ret.append((category, list(floatArray)))

            except Exception as e:
                print(str(e) + f" IGNORING THIS FILE {imageFile}")

        return ret
