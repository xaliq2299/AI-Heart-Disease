import random
import math
import numpy as np

def shuffleTrainingData(dataSet):
    i = len(dataSet)
    while i > 1:
        i = i - 1
        j = random.randrange(i)  # 0 <= j <= i-1
        dataSet[j], dataSet[i] = dataSet[i], dataSet[j]
    return dataSet

def dataNormalizaion(arr):
	# arr = np.array(arr, dtype=np.float32)
	minV = float(min(arr))
	maxV = float(max(arr))
	temp = []
	for i in arr:
		temp.append((float(i)-minV)/(maxV-minV))
	return temp

def initMatrix(row, column):
	M = [[] for i in range(row)]
	for i in range(row):
		for j in range(column):
			M[i].append(random.uniform(-1.,1.))
	return np.array(M, dtype = np.float32)

def createOneHot(k, classes):
	Y = []
	for i in range(classes):
		Y.append(0)
	Y[int(k)] = 1
	return Y

def loadAttributesAndLabels(dataSet, dataIndex, classes, batchSize):
	X = [[] for i in range(batchSize)]
	Y = []
	for i in range(batchSize):
		for j in range(len(dataSet[i])-1):
			X[i].append(dataSet[dataIndex+i][j])
		Y.append(createOneHot(int(dataSet[dataIndex][-1]), classes))
	return np.transpose(X), np.transpose(Y) 

def printMatrix(M):
	for i in range(len(M)):
		for j in range(len(M[i])):
			print(str(M[i][j]) + " ",end="")
		print()

def tanh(Z):
	for i in range(len(Z)):	
		for j in range(len(Z[i])):
			Z[i][j] = (float(2)/(1 + np.exp(float(-2) * Z[i][j]))) - 1
	return Z

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

def hadamard(A, B):
	return np.matrix(np.array(A) * np.array(B))
	# return A*B

def convertProb(Y):
	Y = np.transpose(Y)
	Y = np.array(Y)
	temp = [[] for i in range(len(Y))]
	for i in range(len(Y)):
		maxV = max(Y[i])
		for j in range(len(Y[i])):
			if Y[i][j] == maxV:
				temp[i].append(1)
				continue
			temp[i].append(0)
	return np.transpose(temp)

def compareOutput(A, B):
	# A = np.transpose(A)
	# B = np.transpose(B)
	A = np.array(A)
	B = np.array(B)
	counter = 0
	for i in range(len(A)):
		equal = True
		for j in range(len(A[i])):
			if A[i][j] != B[i][j]:
				equal = False
				break
		if equal:
			counter += 1
	return counter
