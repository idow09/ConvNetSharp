using System;
using System.IO;

namespace FluentMnistDemo
{
    internal class DataSetsLargeImages
    {
        private const string dataFolder = @"../LargeImages/";

        public GenericDataSet Train { get; set; }

        public GenericDataSet Validation { get; set; }

        public GenericDataSet Test { get; set; }

        public bool Load(int validationSize = 1000)
        {
            Directory.CreateDirectory(dataFolder);

            // Load data
            Console.WriteLine("Loading the datasets...");
            var trainImages = LargeImagesReader.Load(Path.Combine(dataFolder, "Train"));
            var testingImages = LargeImagesReader.Load(Path.Combine(dataFolder, "Test"));

            var validationImages = trainImages.GetRange(trainImages.Count - validationSize, validationSize);
            trainImages = trainImages.GetRange(0, trainImages.Count - validationSize);

            if (trainImages.Count == 0 || validationImages.Count == 0 || testingImages.Count == 0)
            {
                Console.WriteLine("Missing training/testing files.");
                Console.ReadKey();
                return false;
            }

            Train = new GenericDataSet(trainImages);
            Validation = new GenericDataSet(trainImages);
            Test = new GenericDataSet(trainImages);
            return true;
        }
    }
}