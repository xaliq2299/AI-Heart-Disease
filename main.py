from NeuralNet import NeuralNet
import numpy as np

def read_file(filename):
	data_from_file = []
	f=open(filename,"r")
	while True:
		temp=f.readline().strip()
		if temp=="":
			break
		data_from_file.append(temp.split(","))
	return data_from_file

def main():
	data_from_file = []
	data_from_file = np.array(read_file("iris_num.data"))
	neuron = NeuralNet(data_from_file, 3, 1, 3)




main()
