using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly List<Matrix<double>> _biases = new List<Matrix<double>>();
        private readonly List<Matrix<double>> _weights = new List<Matrix<double>>();
        private int _layerCount;
        private List<int> _layerSizes;

        public NeuralNetwork(List<int> layerSizes)
        {
            _layerCount = layerSizes.Count;
            _layerSizes = layerSizes;

            foreach (var layerSize in layerSizes.Skip(1))
            {
                _biases.Add(Matrix<double>.Build.Random(layerSize, 1));
            }

            foreach (var pair in layerSizes.TakeAllButLast().Zip(layerSizes.Skip(1), Tuple.Create))
            {
                _weights.Add(Matrix<double>.Build.Random(pair.Item2, pair.Item1));
            }
        }
    }
}