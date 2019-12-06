import NNLib as NN
import numpy as np

'''
!!!!!!!!!!!!!TODO!!!!!!!!!!!
EARLY STOPPING : MAKE TO DO FUNCTION CHECK ACCURACY DIRECTLY AND UPDATE WEIGHTS AND BIOSES
EXPORT CSV FILE 
CHECK tanh FUNCTION 


'''

class NeuralNet(object):
	"""docstring for NeuralNet"""
	data, W1, W2, b1, b2 = [], [], [], [], []
	nbHiddenNodes = 0
	nbBatch = 0
	nbClass = 0
	nbFeatures = 0
	eta = 0.05
	def __init__(self, data, numberOfClasses, batchSize, numberOfHiddenNodes):
		self.data = data
		self.data = self.data.astype(np.float64)
		for i in range(data.shape[1] - 1):
			self.data[:,i] = NN.standartization(self.data[:,i])
		self.nbClass = numberOfClasses
		self.nbBatch = batchSize
		self.nbHiddenNodes = numberOfHiddenNodes
		self.nbFeatures = len(data[0])-1
				
		self.W1 = NN.initMatrix(self.nbHiddenNodes, self.nbFeatures)
		self.W2 = NN.initMatrix(self.nbClass, self.nbHiddenNodes)
		self.b1 = NN.initMatrix(self.nbHiddenNodes, 1) # TODO CHECK LATER!! 
		self.b2 = NN.initMatrix(self.nbClass, 1) # TODO CHECK LATER!! 

		
		self.train(15)
		print()


	def train(self, nbEpoch):
		for i in range(nbEpoch):
			data = NN.shuffleTrainingData(self.data)
			trainData, testData = self.dataSplit(data, 0.7)
			self.trainingEpoch(trainData)
			self.testPrediction(testData)


	def trainingEpoch(self, trainData):
		dataIndex = 0
		batchSize = self.nbBatch
		while True:
			if (dataIndex + batchSize) >= len(trainData):
				if dataIndex >= len(trainData):
					break
				batchSize = len(trainData) - dataIndex 
			X_train, Y_train = NN.loadAttributesAndLabels(trainData, dataIndex, self.nbClass, self.nbBatch, self.nbFeatures)
			dataIndex += batchSize
			# Feed-Forward
			Z1 = self.W1@X_train + self.b1
			A1 = NN.tanh(Z1)
			Z2 = self.W2@A1 + self.b2
			A2 = NN.tanh(Z2)

			# Back-propogation
			error = A2 - Y_train
			delta2 = error*NN.tanhDeriv(A2)
			self.W2 = self.W2 - self.eta * (delta2@np.transpose(A1))
			self.b2 = self.b2 - self.eta * delta2
			delta1 = NN.tanhDeriv(A1)*(np.transpose(self.W2)@delta2)
			self.W1 = self.W1 - self.eta * (delta1@np.transpose(X_train))
			self.b1 = self.b1 - self.eta * delta1

	def testPrediction(self, testData):
		predicted = 0
		dataIndex = 0
		batchSize = self.nbBatch
		while True:
			if (dataIndex + batchSize) >= len(testData):
				if dataIndex >= len(testData):
					break
				batchSize = len(testData) - dataIndex 
			X_train, Y_train = NN.loadAttributesAndLabels(testData, dataIndex, self.nbClass, self.nbBatch, self.nbFeatures)
			dataIndex += batchSize
			# Feed-Forward
			Z1 = self.W1@X_train + self.b1
			A1 = NN.tanh(Z1)
			Z2 = self.W2@A1 + self.b2
			A2 = NN.tanh(Z2)
			A2 = NN.convertProb(A2)

			# Compare result
			predicted += NN.compareOutput(A2, Y_train)
		print('MLP\'s accuracy=' + str(float(predicted/len(testData))*100) + '%')
		return predicted
		

	def dataSplit(self, data, trainCoef):
		trainingSize = int(len(data) * trainCoef)
		trainData = data[:trainingSize][:]
		testData = data[trainingSize:][:]
		return trainData, testData