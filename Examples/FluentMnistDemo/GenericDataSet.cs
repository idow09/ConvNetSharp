using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Volume = ConvNetSharp.Volume.Volume<double>;

namespace FluentMnistDemo
{
    internal class GenericDataSet
    {
        private readonly List<Entry> _trainEntries;
        private readonly Random _random = new Random(RandomUtilities.Seed);
        private int _start;
        private int _epochCompleted;

        public GenericDataSet(List<Entry> trainEntries)
        {
            _trainEntries = trainEntries;
        }

        public Tuple<Volume, Volume, int[]> NextBatch(int batchSize)
        {
            const int w = 400;
            const int h = 225;
            const int numClasses = 10;

            var dataShape = new Shape(w, h, 1, batchSize);
            var labelShape = new Shape(1, 1, numClasses, batchSize);
            var data = new double[dataShape.TotalLength];
            var label = new double[labelShape.TotalLength];
            var labels = new int[batchSize];

            // Shuffle for the first epoch
            if (_start == 0 && _epochCompleted == 0)
            {
                for (var i = _trainEntries.Count - 1; i >= 0; i--)
                {
                    var j = _random.Next(i);
                    var temp = _trainEntries[j];
                    _trainEntries[j] = _trainEntries[i];
                    _trainEntries[i] = temp;
                }
            }

            var dataVolume = BuilderInstance.Volume.From(data, dataShape);

            for (var i = 0; i < batchSize; i++)
            {
                var entry = _trainEntries[_start];

                labels[i] = entry.Label;

                var j = 0;
                for (var y = 0; y < h; y++)
                {
                    for (var x = 0; x < w; x++)
                    {
                        dataVolume.Set(x, y, 0, i, entry.Image[j++] / 255.0);
                    }
                }

                label[i * numClasses + entry.Label] = 1.0;

                _start++;
                if (_start == _trainEntries.Count)
                {
                    _start = 0;
                    _epochCompleted++;
                    Console.WriteLine($"Epoch #{_epochCompleted}");
                }
            }


            var labelVolume = BuilderInstance.Volume.From(label, labelShape);

            return new Tuple<Volume, Volume, int[]>(dataVolume, labelVolume, labels);
        }
    }
}