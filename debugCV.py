import numpy as np
from decisiontree import DTClassifier
from arff import Arff
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sklearn.model_selection import KFold,train_test_split,cross_val_score

mat = Arff("./datasets/lenses.arff")

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
    counts += [mat.unique_value_count(i)]
X = mat.data[:,0:-1]
y = mat.data[:, -1].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=3)
DTClass = DTClassifier(counts)

kfold = KFold(n_splits=10, random_state=18)
results = cross_val_score(DTClass, X_train, y_train, cv=kfold)
print(results)
print(np.mean(results))
