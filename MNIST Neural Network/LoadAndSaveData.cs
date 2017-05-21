using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural_Network
{
    class LoadAndSaveData
    {
        private static LoadAndSaveData instance;

        private LoadAndSaveData()
        {
        }

        public static LoadAndSaveData GetInstance()
        {
            if (instance == null)
                instance = new LoadAndSaveData();
      
            return instance;
        }

        public void SaveNS(int epochs, NeuralNetwork NN)
        {
            StreamWriter fout = new StreamWriter("weight" + epochs + ".txt");

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

        public void LoadNS(NeuralNetwork NN)
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

        public NumberImage LoadNumberImage(int num, int order)
        {
            string nameFile = @".\Numbers\" + num + "" + order + ".png";
            var png = (Bitmap)Image.FromFile(nameFile);

            double[][] pixels = new double[png.Width][];
            for (int x = 0; x < png.Width; x++)
            {
                pixels[x] = new double[png.Height];
                for (int y = 0; y < png.Height; y++)
                {
                    pixels[x][y] = (255 - (png.GetPixel(y, x).R * 0.299 + png.GetPixel(y, x).G * 0.587 + png.GetPixel(y, x).B * 0.114)) / 255;
                }
            }

            return new NumberImage(png.Height, png.Width, pixels, num);
        }

        public NumberImage[] LoadDataMNIST(string pixelFile, string labelFile, int numImages = 1000)
        {
            NumberImage[] result = new NumberImage[numImages];

            FileStream pixelsStream = new FileStream(pixelFile, FileMode.Open);
            FileStream labelsStream = new FileStream(labelFile, FileMode.Open);
            BinaryReader brImages = new BinaryReader(pixelsStream);
            BinaryReader brLabels = new BinaryReader(labelsStream);

            var magicNumber1 = brImages.ReadInt32();
            var numberoOfImages = brImages.ReadInt32();
            var numberOfRows = brImages.ReadInt32();
            int numberOfColumns = brImages.ReadInt32();
            int magicNumber2 = brLabels.ReadInt32();
            int numberOfItems = brLabels.ReadInt32();

            double[][] pixels = new double[28][];
            for (int i = 0; i < pixels.Length; ++i)
            {
                pixels[i] = new double[28];
            }

            for (int k = 0; k < numImages; ++k)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        double byteOnPixel = brImages.ReadByte();
                        pixels[i][j] = byteOnPixel / 255;
                    }
                }
                byte label = brLabels.ReadByte();
                NumberImage dImage = new NumberImage(28, 28, pixels, label);
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
