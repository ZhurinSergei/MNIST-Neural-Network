using System;
using System.Linq;

namespace MNIST_Neural_Network
{
    class Neuron
    {
        private double[] weight;
        private double bias;
        private int numberOfSynapses;
        private double error;

        private Random rand;

        public Neuron(int numberOfSynapses, Random rand)
        {
            this.rand = rand;
            this.numberOfSynapses = numberOfSynapses;
            this.weight = new Double[numberOfSynapses];
            this.bias = 1;
            RandomizeWeights();
        }

        public double[] GetWeight()
        {
            return weight;
        }

        /// <summary>
        /// t - ожидаемый выход нейрона
        /// o - действительный выход нейрона
        /// </summary>
        /// <param name="t"></param>
        /// <param name="o"></param>
        public void UpdateErrorOutputLayer(double target, double output)
        {
            error = (target - output) * output * (1 - output);
        }

        public void UpdateError(Neuron[] neurons, double output, int magicNumber)
        {
            double summ = 0;

            for (int i = 0; i < neurons.Count(); i++)
            {
                summ += neurons[i].weight[magicNumber] * neurons[i].error;
            }
            error = output * (1 - output) * summ;
        }

        /// <summary>
        /// Возвращает значение выхода нейрона
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double OutputOfANeuron(double[] input)
        {
            double power = 0;
            for (int r = 0; r < numberOfSynapses; r++)
            {
                power += weight[r] * input[r];
            }
            power += bias * 1;

            return Sigmoid(power);
        }

        public void UpdateWeights(double[] output)
        {
            double epsilone = 0.5;

            for (int r = 0; r < numberOfSynapses; r++)
            {
                weight[r] += epsilone * error * output[r];
            }
            bias += epsilone * error * 1; // 1 - constant input for any bias;

        }

        private void RandomizeWeights()
        {
            for (int r = 0; r < numberOfSynapses; r++)
            {
                weight[r] = (double)rand.Next(-10000, 10001) / 20000; // random [-0.5; 0.5]
            }
        }

        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        private double Sigmoid(Double X)
        {
            return 1 / (1 + Math.Pow(Math.E, -1 * X));
        }
    }
}