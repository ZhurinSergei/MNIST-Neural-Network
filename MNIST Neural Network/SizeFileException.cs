using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural_Network
{
    class SizeFileException : ApplicationException
    {
        public SizeFileException()
            : base("Not the right size of the file.")
        { }
    }
}