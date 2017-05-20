using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural_Network
{
    class NeuralNetwork
    {
        private Neuron[][] neurons; // [Количество слоев][количество нейронов в слое]
        private Random rand = new Random();

        public NeuralNetwork(int[] countNeuronsInLayers)
        {
            neurons = new Neuron[countNeuronsInLayers.Count()][];

            for (int i = 0; i < countNeuronsInLayers.Count(); i++)
            {
                neurons[i] = new Neuron[countNeuronsInLayers[i]];
                if (i != 0)
                {
                    for (int j = 0; j < countNeuronsInLayers[i]; j++)
                    {
                        neurons[i][j] = new Neuron(countNeuronsInLayers[i - 1], rand);
                    }
                }
            }
        }

        public Neuron[][] GetNeurons()
        {
            return neurons;
        }

        /// <summary>
        /// Возвращаетмассив массивов значений выходов нейронов
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double[][] OutputOfANeurons(double[] input)
        {
            double[][] output = new double[neurons.Length][];

            for (int i = 1; i < output.Length; i++)
            {
                output[i] = new double[neurons[i].Count()];

                if (i == 1)
                {
                    for (int j = 0; j < neurons[i].Count(); j++)
                    {
                        output[i][j] = neurons[i][j].OutputOfANeuron(input);
                    }
                }
                else
                {
                    for (int j = 0; j < neurons[i].Count(); j++)
                    {
                        output[i][j] = neurons[i][j].OutputOfANeuron(output[i - 1]);
                    }
                }
            }
            return output;
        }

        /// <summary>
        /// Возвращает номер нейрона с максимальным выходом
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double GetAnswer(double[] input)
        {
            double[][] output = OutputOfANeurons(input);
            double max = int.MinValue;
            int number = -1;

            for (int i = 0; i < output[neurons.Count() - 1].Count(); i++)
            {
                if (output[neurons.Count() - 1][i] > max)
                {
                    max = output[neurons.Count() - 1][i];
                    number = i;
                }
            }
            
            return number;
        }

        /// <summary>
        /// если пример был выучен, метод вернет true, иначе он его выучит
        /// </summary>
        /// <param name="input"></param>
        /// <param name="correctOutput"></param>
        /// <returns></returns>
        public bool Study(double[] input, double[] correctOutput)
        {
            double[][] output = OutputOfANeurons(input);
            int countlearning = 0;

            if (CompareVectors(correctOutput, output[neurons.Count() - 1]))
            {
                return true;
            }

            while (!CompareVectors(correctOutput, output[neurons.Count() - 1]))
            {
                countlearning++;
                if (countlearning > 1000)
                {
                    break;
                }

                for (int i = 0; i < output[neurons.Count() - 1].Count(); i++)
                {
                    neurons[neurons.Count() - 1][i].UpdateErrorOutputLayer(correctOutput[i], output[neurons.Count() - 1][i]);
                }

                for (int i = output.Count() - 2; i >= 1; i--)
                {
                    for (int j = 0; j < output[i].Count(); j++)
                    {
                        neurons[i][j].UpdateError(neurons[i + 1], output[i][j], j);
                    }
                }

                for (int i = neurons.Count() - 1; i >= 1; i--)
                {
                    for (int j = 0; j < neurons[i].Count(); j++)
                    {
                        if (i - 1 != 0)
                        {
                            neurons[i][j].UpdateWeights(output[i - 1]);
                        }
                        else if (i - 1 == 0)
                        {
                            neurons[i][j].UpdateWeights(input);
                        }
                    }
                }
                output = OutputOfANeurons(input);
            }

            return false;
        }

        private bool CompareVectors(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
                if ((a[i] == 0 && b[i] > 0.1) || (a[i] == 1 && b[i] < 0.9))
                    return false;

            return true;
        }
    }
}