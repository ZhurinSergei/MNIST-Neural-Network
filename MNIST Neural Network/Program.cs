using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural_Network
{
    class Program
    {
        private static NeuralNetwork NN = new NeuralNetwork(new int[] { 28 * 28, 500, 150, 10 });
        private static NumberImage[] input;
        private static NumberImage[] test;

        private static double[][] correctOutput;
        private static int epochs = 0;

        static void Main(string[] args)
        {
            string pixelFile = "train-images.idx3-ubyte";
            string labelFile = "train-labels.idx1-ubyte";
            string pixelTest = "t10k-images.idx3-ubyte";
            string labelTest = "t10k-labels.idx1-ubyte";

            input = LoadDataMNIST(pixelFile, labelFile, 60000);
            test = LoadDataMNIST(pixelTest, labelTest, 10000);

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

            //      StartStudy();
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

                //SaveNS(epochs, (double)h / 1000);
            } while (!b);
        }

        private static void Example()
        {
            int h = 0;
            LoadNS();

            for (int j = 1; j < 6; j++)
            {
                for (int l = 0; l < 10; l++)
                {
                    double[] number = LoadNumberImage(l + "" + j + ".png");

                    Console.WriteLine(l + "" + j + ".png = " + NN.GetAnswer(number));
                    if (l == NN.GetAnswer(number)) h++;
                }
                Console.WriteLine();
            }
            Console.WriteLine((double)h / 50 * 100 + "%\n");

        }

        /// <summary>
        /// Имя растрового изображения 28*28 пикселей в котором нарисована цифра
        /// Возвращает массив double от 0 до 1, где 1 черный цвет пикселя, 0 белый
        /// </summary>
        /// <param name="nameFile"></param>
        /// <returns></returns>
        private static double[] LoadNumberImage(string nameFile)
        {
            var png = (Bitmap)Image.FromFile(nameFile);
            double[] number = new double[28 * 28];

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    number[x * 28 + y] = (255 - (png.GetPixel(y, x).R * 0.299 + png.GetPixel(y, x).G * 0.587 + png.GetPixel(y, x).B * 0.114)) / 255;
                }
            }

            return number;
        }

        private static void DigitRecognitionMNIST()
        {
            Console.WriteLine("Wait");
            int h = 0;
            for (int i = 0; i < 10000; i++)
            {
                if (NN.GetAnswer(test[i].GetPixels()) == test[i].GetLabel())
                {
                    h++;
                }
            }
            Console.WriteLine("Точность распознования выборки MNIST = {0} \n", ((double)h / 10000) * 100 + "%");
        }

        private static void SaveNS(int epochs, double h)
        {
            StreamWriter fout = new StreamWriter("weight" + epochs + "B" + h + ".txt");

            for (int i = 1; i < NN.GetNeurons().Count(); i++)
            {
                for (int j = 0; j < NN.GetNeurons()[i].Count(); j++)
                {
                    for (int k = 0; k < NN.GetNeurons()[i][j].GetWeight().Count(); k++)
                    {
                        fout.WriteLine(NN.GetNeurons()[i][j].GetWeight()[k] + " ");
                    }
                }

            }

            fout.Close();
        }

        private static void LoadNS()
        {
            StreamReader fin = new StreamReader("weight13B9,833.txt"); //2 скрытых слоя 500, 150

            for (int i = 1; i < NN.GetNeurons().Count(); i++)
            {
                for (int j = 0; j < NN.GetNeurons()[i].Count(); j++)
                {
                    for (int k = 0; k < NN.GetNeurons()[i][j].GetWeight().Count(); k++)
                    {
                        NN.GetNeurons()[i][j].GetWeight()[k] = double.Parse(fin.ReadLine());
                    }
                }
            }

            fin.Close();
        }

        private static NumberImage[] LoadDataMNIST(string pixelFile, string labelFile, int numImages = 1000)
        {
            NumberImage[] result = new NumberImage[numImages];

            double[][] pixels = new double[28][];
            for (int i = 0; i < pixels.Length; ++i)
            {
                pixels[i] = new double[28];
            }

            FileStream pixelsStream = new FileStream(pixelFile, FileMode.Open);
            FileStream labelsStream = new FileStream(labelFile, FileMode.Open);
            BinaryReader brImages = new BinaryReader(pixelsStream);
            BinaryReader brLabels = new BinaryReader(labelsStream);

            int magicNumber1 = brImages.ReadInt32();
            int numberoOfImages = brImages.ReadInt32();
            int numberOfRows = brImages.ReadInt32();
            int numberOfColumns = brImages.ReadInt32();
            int magicNumber2 = brLabels.ReadInt32();
            int numberOfItems = brLabels.ReadInt32();

            for (int k = 0; k < numImages; ++k)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        double b = brImages.ReadByte();
                        pixels[i][j] = b / 255;
                    }
                }
                byte label = brLabels.ReadByte();
                NumberImage dImage = new NumberImage(pixels, label);
                result[k] = dImage;
            }
            pixelsStream.Close();
            labelsStream.Close();
            brImages.Close();
            brLabels.Close();
            return result;
        }
    }
}