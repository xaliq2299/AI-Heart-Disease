import NNLib as NN

'''
!!!!!!!!!!!!!TODO!!!!!!!!!!!
EARLY STOPPING : MAKE TO DO FUNCTION CHECK ACCURACY DIRECTLY AND UPDATE WEIGHTS AND BIOSES
EXPORT CSV FILE 



'''

class NeuralNet(object):
	"""docstring for NeuralNet"""
	data, W1, W2, b1, b2 = [], [], [], [], []
	nbHiddenNodes = 0
	nbBatch = 0
	nbClass = 0
	nbFeatures = 0
	eta = 0.01
	def __init__(self, data, numberOfClasses, batchSize, numberOfHiddenNodes):
		self.data = data
		self.nbClass = numberOfClasses
		self.nbBatch = batchSize
		self.nbHiddenNodes = numberOfHiddenNodes
		self.nbFeatures = len(data[0])-1
		data = NN.shuffleTrainingData(data)
		trainData, testData = self.dataSplit(data, 0.7)

		
		self.W1 = NN.initMatrix(self.nbHiddenNodes, self.nbFeatures)
		self.W2 = NN.initMatrix(self.nbClass, self.nbHiddenNodes)
		self.b1 = NN.initMatrix(self.nbHiddenNodes, 1) # TODO CHECK LATER!! 
		self.b2 = NN.initMatrix(self.nbClass, 1) # TODO CHECK LATER!! 

		print(str(self.W1))
		print()
		print(str(self.W2))
		print()
		print(str(self.b1))
		print()
		print(str(self.b2))
		print()


	# def train(self, nbEpoch):
	# 	for i in range(nbEpoch):
			

	# def trainingEpoch():
		

	def dataSplit(self, data, trainCoef):
		trainingSize = int(len(data) * trainCoef)
		# testSize = len(data) - trainingSize
		trainData = data[:trainingSize][:]
		testData = data[trainingSize:][:]
		return trainData, testData