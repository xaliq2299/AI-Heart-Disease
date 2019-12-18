import pandas as pd
from DTLib import DT
import DTLib as DTLib

def main():
	data = pd.read_csv('jouer.csv', sep=';')

	# data, edge_values = DTLib.transfer_values(data, [0, 3, 4, 7, 9], 4)
	data, edge_values = DTLib.transfer_values(data, [1, 2], 3)
	print(str(data)+'\n')
	root = DT(None)
	visited = []
	root = DTLib.decision_tree(root, data, visited, 1)
	print('\n\n\n\n\n\n\n')
	DTLib.print_tree(root)

main()


