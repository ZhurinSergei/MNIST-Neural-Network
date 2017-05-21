namespace MNIST_Neural_Network
{
    class NumberImage
    {
        private int width;
        private int height;

        private int size;
        private double[] pixels;
        private int label;

        public NumberImage(int numberOfRows, int numberOfColumns, double[][] pixels, int label)
        {
            if (numberOfColumns != numberOfRows) throw new SizeFileException();
            this.size = numberOfRows;

            this.width = numberOfColumns;
            this.height = numberOfRows;
            this.pixels = new double[width * height];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    this.pixels[i * height + j] = pixels[i][j];
                }
            }

            this.label = label;
        }

        public int GetLabel()
        {
            return label;
        }

        public double[] GetPixels()
        {
            return pixels;
        }

        public int GetSize()
        {
            return size;
        }
    }
}