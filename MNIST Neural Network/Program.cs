using System;

namespace MNIST_Neural_Network
{
    static class Program
    {
        private static NeuralNetwork NN;
        private static NumberImage[] input;
        private static NumberImage[] test;
        private static double[][] correctOutput;
        private static int epochs = 0;

        static void Main(string[] args)
        {
            LoadAndSaveData dataWork = LoadAndSaveData.GetInstance();
            NN = new NeuralNetwork(new int[] { 28 * 28, 500, 150, 10 });

            string pixelFile = "train-images.idx3-ubyte";
            string labelFile = "train-labels.idx1-ubyte";
            string pixelTest = "t10k-images.idx3-ubyte";
            string labelTest = "t10k-labels.idx1-ubyte";

            input = dataWork.LoadDataMNIST(pixelFile, labelFile, 60000);
            test = dataWork.LoadDataMNIST(pixelTest, labelTest, 10000);

            correctOutput = new double[60000][];
            for (int i = 0; i < 60000; i++)
            {
                correctOutput[i] = new double[10];
            }

            for (int i = 0; i < 60000; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    correctOutput[i][j] = 0;
                    correctOutput[i][input[i].GetLabel()] = 1;
                }
            }

            //         StartStudy();
            Example();       // Example of the work of a neural network on real data (scanned images of numbers)

            DigitRecognitionMNIST();

        }

        private static void StartStudy()
        {
            bool b;
            do
            {
                epochs++;

                b = true;
                for (int i = 0; i < 60000; i++)
                {
                    b &= NN.Study(input[i].GetPixels(), correctOutput[i]);

                    if (i % 100 == 0) Console.WriteLine("Количество выученых примеров: - " + i);
                }

                DigitRecognitionMNIST();

                Console.WriteLine("Epochs = " + epochs);
                Console.WriteLine("----------------------");

            } while (!b);
        }

        private static void Example()
        {
            NumberImage[] number = new NumberImage[50];
            int countOfCorrectlyRecognized = 0;

            LoadAndSaveData.GetInstance().LoadNS(NN);

            for (int order = 0; order < 5; order++)
            {
                for (int num = 0; num < 10; num++)
                {
                    number[order * 10 + num] = LoadAndSaveData.GetInstance().LoadNumberImage(num, order);

                    int answer = NN.GetAnswer(number[order * 10 + num].GetPixels());

                    Console.WriteLine(order + "" + num + ".png = " + answer);
                    if (num == answer) countOfCorrectlyRecognized++;
                }
                Console.WriteLine();
            }

            Console.WriteLine((double)countOfCorrectlyRecognized / 50 * 100 + "%\n");
        }

        private static void DigitRecognitionMNIST()
        {
            Console.WriteLine("Wait");
            int countOfCorrectlyRecognized = 0;

            for (int i = 0; i < 10000; i++)
            {
                if (NN.GetAnswer(test[i].GetPixels()) == test[i].GetLabel())
                {
                    countOfCorrectlyRecognized++;
                }
            }
            Console.WriteLine("Точность распознования выборки MNIST = {0} \n", ((double)countOfCorrectlyRecognized / 10000) * 100 + "%");
        }
    }
}