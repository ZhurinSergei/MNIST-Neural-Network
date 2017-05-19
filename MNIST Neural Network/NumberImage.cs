using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural_Network
{
    class NumberImage
    {
        public int width = 28;
        private int height = 28;
        private double[] pixels;
        private byte label;

        public NumberImage(double[][] pixels, byte label)
        {
            this.pixels = new double[28 * 28];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    this.pixels[i * height + j] = pixels[i][j];
                }
            }

            this.label = label;
        }

        public byte GetLabel()
        {
            return label;
        }

        public double[] GetPixels()
        {
            return pixels;
        }
    }
}