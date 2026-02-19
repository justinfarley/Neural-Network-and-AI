import numpy as np
class ImageProcessing:

    @staticmethod
    def GetPixelColorsAsFloatArray(bitmap, targetWidth=32, targetHeight=32):

        # Resize image
        resized = bitmap.resize((targetWidth, targetHeight))

        # Ensure RGB format (equivalent to Format24bppRgb)
        resized = resized.convert("RGB")

        # Convert to numpy array
        pixels = np.array(resized)

        # Normalize to [0,1] and flatten
        float_array = pixels.astype(np.float32) / 255.0

        # Flatten to 1D array in R,G,B order
        return float_array.flatten()
