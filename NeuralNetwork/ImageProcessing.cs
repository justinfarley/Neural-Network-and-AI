    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;

    namespace Neural_Network_and_AI
    {
        public static class ImageProcessing
        {
            public static float[] GetPixelColorsAsFloatArray(Bitmap bitmap, int targetWidth = 32, int targetHeight = 32)
            {
                var resized = new Bitmap(bitmap, new Size(targetWidth, targetHeight));
                
                var data = resized.LockBits(new Rectangle(0, 0, resized.Width, resized.Height),
                    ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                try
                {
                    int byteCount = data.Stride * data.Height;
                    byte[] pixels = new byte[byteCount];
                    Marshal.Copy(data.Scan0, pixels, 0, byteCount);

                    float[] floatArray = new float[resized.Width * resized.Height * 3];
                    int floatIndex = 0;
                    for (int row = 0; row < resized.Height; row++)
                    {
                        int rowStart = row * data.Stride;
                        for (int col = 0; col < resized.Width; col++)
                        {
                            int pixelStart = rowStart + col * 3;
                            floatArray[floatIndex++] = pixels[pixelStart + 2] / 255.0f; // R
                            floatArray[floatIndex++] = pixels[pixelStart + 1] / 255.0f; // G
                            floatArray[floatIndex++] = pixels[pixelStart] / 255.0f;     // B
                        }
                    }

                    return floatArray;
                }
                finally
                {
                    resized.UnlockBits(data);
                    resized.Dispose();
                }
            }   
        }
    }