﻿using System;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Fluent;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Core.Serialization;
using Volume = ConvNetSharp.Volume.Volume<double>;

namespace FluentMnistDemo
{
    internal class Program
    {
        private readonly CircularBuffer<double> _testAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _trainAccWindow = new CircularBuffer<double>(100);
        private INet<double> _net;
        private int _stepCount;
        private SgdTrainer<double> _trainer;

        private static void Main()
        {
            var program = new Program();
            program.MnistDemo();
        }

        private void MnistDemo()
        {
            var datasets = new DataSetsLargeImages();
            if (!datasets.Load(5))
            {
                return;
            }

            // Create network
            this._net = FluentNet<double>.Create(400, 225, 1)
                .Conv(5, 5, 8).Stride(1).Pad(2)
                .Relu()
                .Pool(2, 2).Stride(2)
                .Conv(5, 5, 16).Stride(1).Pad(2)
                .Relu()
                .Pool(3, 3).Stride(3)
                .FullyConn(10)
                .Softmax(10)
                .Build();

            this._trainer = new SgdTrainer<double>(this._net)
            {
                LearningRate = 0.01,
                BatchSize = 2,
                Momentum = 0.9
            };

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            do
            {
                var trainSample = datasets.Train.NextBatch(this._trainer.BatchSize);
                Train(trainSample.Item1, trainSample.Item2, trainSample.Item3);

                var testSample = datasets.Test.NextBatch(this._trainer.BatchSize);
                Test(testSample.Item1, testSample.Item3, this._testAccWindow);

                Console.WriteLine("Loss: {0} Train accuracy: {1}% Test accuracy: {2}%", this._trainer.Loss,
                    Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2),
                    Math.Round(this._testAccWindow.Items.Average() * 100.0, 2));

                Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this._stepCount,
                    Math.Round(this._trainer.ForwardTimeMs, 2),
                    Math.Round(this._trainer.BackwardTimeMs, 2));
                    if (this._stepCount > 60000)
                    {
                        break;
                    }
            } while (!Console.KeyAvailable);
            
            // Serialize to json 
            var json = ((FluentNet<double>) this._net).ToJson(); 
            System.IO.File.WriteAllText(@"../mnist_fluent.json", json);
            
            Console.WriteLine("Network Saved");
        }

        private void Test(Volume x, int[] labels, CircularBuffer<double> accuracy, bool forward = true)
        {
            if (forward)
            {
                this._net.Forward(x);
            }

            var prediction = this._net.GetPrediction();

            for (var i = 0; i < labels.Length; i++)
            {
                accuracy.Add(labels[i] == prediction[i] ? 1.0 : 0.0);
            }
        }

        private void Train(Volume x, Volume y, int[] labels)
        {
            this._trainer.Train(x, y);

            Test(x, labels, this._trainAccWindow, false);

            this._stepCount += labels.Length;
        }
    }
}