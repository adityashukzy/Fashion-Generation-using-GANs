"""
Main Driver Code
"""

from ML.newTrain import Train
import matplotlib.pyplot as plt
from time import time

def main(inp_path, inp_epochs, inp_split):
	model_train = Train(path=inp_path, epochs=inp_epochs, batch_size=4, split=inp_split, noise_dim=100)
	starting = time()
	
	model_train.trainer()
	print(time() - starting)
	model_train.plot_trainer()

if __name__ == "__main__":
	main()
