using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;

namespace Neural_Network_and_AI
{
    public interface Dataset
    {
        /// <param name="nn">The Neural Network to test on</param>
        public void Testing(NeuralNetwork nn);

        /// <summary>
        /// 
        /// </summary>
        /// <returns>a tuple with item 1 = all of the input float arrays, item 2 = all of the expected outputs for said array</returns>
        public (List<List<float>>, List<List<float>>) TrainingData(int samples = -1);
    
        /// <summary>
        /// NOTE: MUST have every category in "categories" array to match the folder names of each category. If you want to Load per-folder, 
        /// use <see cref="LoadDataFromFile"/>
        /// </summary>
        /// <param name="path"></param>
        /// <param name="categories"></param>
        /// <param name="extension"></param>
        /// <param name="imageProcessingFunction"></param>
        /// <returns></returns>
        public static List<Tuple<string, List<float>>> LoadDataFromFileAll(string path, string extension, Func<Bitmap, int, int, float[]> imageProcessingFunction, int compressionX, int compressionY, string[] categories, int samples)
        {
            List<Tuple<string, List<float>>> ret = new List<Tuple<string, List<float>>>();
            int samplesPerCategory = -1;
            if(samples > 0)
                samplesPerCategory = samples / categories.Length;
            foreach(var category in categories)
            {
                var files = Directory.GetFiles($"{path}/{category}", $"*{extension}");
                Console.WriteLine($"Directory: {path}/{category}, Files found: {files.Length}");
                int count = 0;
                foreach(var imageFile in files)
                {
                    count++;
                    if(samples > 0 && count > samplesPerCategory) 
                    {
                        Console.WriteLine($"Directory: {path}/{category}, Files loaded in sample: {ret.Count}/{samples}");
                        break;
                    }
                    try
                    {
                        Bitmap bitmap = new Bitmap(imageFile);
                        var floatArray = imageProcessingFunction?.Invoke(bitmap, compressionX, compressionY);

                        //Store category and corresponding pixel float data for use later
                        ret.Add(new Tuple<string, List<float>>(category, floatArray.ToList()));
                    }
                    catch (Exception e)
                    {
                        //Can be ignored, don't add troublesome files
                        Console.WriteLine(e.Message + $"IGNORING THIS FILE {imageFile}");
                    }
                }
            }
            return ret;
        }

        public static List<Tuple<string, List<float>>> LoadDataFromFile(string path, string category, string extension, Func<Bitmap, int, int, float[]> imageProcessingFunction, int compressionX, int compressionY)
        {
            List<Tuple<string, List<float>>> ret = new List<Tuple<string, List<float>>>();

            var files = Directory.GetFiles($"{path}/{category}", $"*{extension}");
            Console.WriteLine($"Directory: {category}, Files found: {files.Length}");
            foreach(var imageFile in files)
            {
                try
                {
                    Bitmap bitmap = new Bitmap(imageFile);
                    var floatArray = imageProcessingFunction?.Invoke(bitmap, compressionX, compressionY);

                    //Store category and corresponding pixel float data for use later
                    ret.Add(new Tuple<string, List<float>>(category, floatArray.ToList()));
                }
                catch (Exception e)
                {
                    //Can be ignored, don't add troublesome files
                    Console.WriteLine(e.Message + $"IGNORING THIS FILE {imageFile}");
                }
            }
            return ret;
        }

    }
}