using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly List<Vector<double>> _biases = new List<Vector<double>>();
        private readonly List<Matrix<double>> _weights = new List<Matrix<double>>();
        private int _layerCount;
        private List<int> _layerSizes;

        public NeuralNetwork(List<int> layerSizes)
        {
            _layerCount = layerSizes.Count;
            _layerSizes = layerSizes;

            foreach (var layerSize in layerSizes.Skip(1))
            {
                _biases.Add(Vector<double>.Build.Random(layerSize, 1));
            }

            foreach (var pair in layerSizes.TakeAllButLast().Zip(layerSizes.Skip(1), Tuple.Create))
            {
                // This might be messed up
                _weights.Add(Matrix<double>.Build.Random(pair.Item2, pair.Item1));
            }
        }

        public Vector<double> FeedForward(Vector<double> networkInput)
        {
            for (int i = 0; i < _biases.Count; i++)
            {
                var bias = _biases[i];
                var weight = _weights[i];

                networkInput = ApplySigmoid(weight * networkInput + bias);
            }

            return networkInput;
        }

        private Vector<double> ApplySigmoid(Vector<double> z)
        {
            z.Map(SpecialFunctions.Logistic, z);
            return z;
        }
    }
}