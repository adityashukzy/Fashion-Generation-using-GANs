"""
Main Driver Code
Modyfying TO run on COLAB - PARAMS

"""

from ML.train import train 

import matplotlib.pyplot as plt


def main(g_path,n_epochs):
    modeltrain = train(path=g_path, epochs=n_epochs, batch_size=4,
                     vec_shape=100, split=[1,2000,0], noisedim=100)
    modeltrain.trainer()

if __name__ == "__main__":
    main()
