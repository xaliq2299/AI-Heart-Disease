import pandas as pd
from DTLib import DT
import DTLib as DTLib

def main():
	pd.set_option('display.max_rows', None)
	data = pd.read_csv('heart_disease_dataset.csv', sep=';')
	data, edge_values = DTLib.transfer_values(data, [0, 3, 4, 7, 9], 4)


	data = data.sample(frac=1)


	root = DT(None)
	root = DTLib.decision_tree(root, data.iloc[:int(len(data)*0.7), :], [], 1, len(data.columns)-1)
	# DTLib.print_tree(root)
	for i in range(len(data)):
		print(str(DTLib.find_result(data.iloc[i, :], root)))







main()
