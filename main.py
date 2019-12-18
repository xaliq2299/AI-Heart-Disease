from NeuralNet import NeuralNet
import numpy as np

def read_file(filename, separator, headerExist = False):
	data_from_file = []
	f=open(filename,"r")
	if headerExist:
		temp=f.readline().strip()
	while True:
		temp=f.readline().strip()
		if temp=="":
			break
		data_from_file.append(temp.split(separator))
	return data_from_file


def main():
	data_from_file = np.array(read_file("heart_disease_dataset.csv", ';', True))
	neuron = NeuralNet(data_from_file, 2, 4, 5)
	print("Enter \"r\" for repeating tests(15 epochs)")
	option = input()
	while option == 'r':
		neuron.train(15)
		print("\nEnter \"r\" for repeating tests(15 epochs)")
		option = input()

	print('Best result achieved during tests:' + str(neuron.bestResult))


main()