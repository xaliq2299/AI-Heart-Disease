import random
import math
import numpy as np

def shuffleTrainingData(dataSet):
	indexes = np.random.permutation(dataSet.shape[0])
	return dataSet[indexes]

def dataNormalization(arr):
	return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def standartization(arr):
	return (arr - np.mean(arr)) / np.std(arr)

def initMatrix(row, column):
	M = [[] for i in range(row)]
	for i in range(row):
		for j in range(column):
			M[i].append(random.uniform(-.0001,.0001))
	return np.array(M, dtype = np.float32)

def loadAttributesAndLabels(dataSet, dataIndex, classes, batchSize, features):
	X = np.zeros([features, batchSize])
	Y = np.zeros([classes, batchSize])
	for i in range(batchSize):
		for j in range(features):
			X[j][i] = dataSet[dataIndex+i][j]
		Y[int(dataSet[dataIndex+i][-1])][i] = 1
	return X, Y


def tanh(Z):
	return (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)

def tanhDeriv(Z):
	return np.subtract(1, np.power(Z,2))

def relu(Z):
	for i in range(len(Z)):
		for j in range(len(Z[i])):
			if Z[i][j] < 0:
				Z[i][j] = 0
	return Z

def reluDeriv(Z):
	for i in range(len(Z)):
		for j in range(len(Z[i])):
			if Z[i][j] > 0:
				Z[i][j] = 1
			else:
				Z[i][j] = 0
	return Z

def floatMultMatrix(A, n):
	for i in range(len(A)):
		for j in range(len(A[i])):
			A[i][j] = A[i][j] * n
	return A

def convertProb(Y):
	Y = np.transpose(Y)
	for i in range(len(Y)):
		maxV = max(Y[i])
		for j in range(len(Y[i])):
			if Y[i][j] == maxV:
				Y[i][j] = 1
				continue
			Y[i][j] = 0
	return np.transpose(Y)

def compareOutput(A, B):
	A = np.transpose(A)
	B = np.transpose(B)
	counter = 0
	for i in range(len(A)):
		if np.array_equal(A[i],B[i]):
			counter += 1
	return counter


def crossEntropy(yHat, y):
	cost = 0.
	for i in range(len(y)):
		for j in range(len(y[0])):
			cost += y[i][j]*np.log(yHat[i][j])
	return -(cost/len(y[0]))

def calcSumOfB(M, batchSize):
	M = np.sum(M, axis=1)/batchSize
	temp = np.zeros((len(M),1))
	for i in range(len(M)):
		temp[i][0] = M[i]
	return temp