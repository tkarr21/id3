import numpy as np
from decisiontree import DTClassifier
from arff import Arff
import pprint
pp = pprint.PrettyPrinter(indent=4)


mat = Arff("./datasets/lenses.arff")

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
    counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:, -1].reshape(-1, 1)
#print(data)
#print(labels)

DTClass = DTClassifier(counts)
DTClass.fit(data, labels)
pp.pprint(DTClass.tree)
print("PRINTED TREE")


