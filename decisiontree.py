import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score


class DTClassifier(BaseEstimator, ClassifierMixin):
    

    def __init__(self, counts):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            DT  = DTClassifier()
        """
        self.counts = counts        # the debug code from learn suite includes the output as a count
        self.features = [i for i in range(len(counts) - 1)]
        self.tree = {}
        

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        #check for nan
        if np.isnan(np.sum(X)):
            
            
            
            indices = np.where(np.isnan(X))
            X[indices] = np.take(self.counts[0:-1], indices[1])


        insta = [i for i in range(X.shape[0])]


        self.tree = self.id3(X, self.features, y, insta, None, None)

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.eval(X[i, :], self.tree))

        return np.array(predictions)


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        preds = self.predict(X)
        #print(f"predictions: {preds}")
        hits = 0
        for i in range(preds.shape[0]):
            if preds[i] == y[i]:
                hits += 1

        return hits / y.shape[0]

    def entropy(self, labels):
        """ calculate entropy of a set
        """
        vals, totals = np.unique(labels, return_counts=True)
        return np.sum([(-totals[i] / np.sum(totals)) * np.log2(totals[i] / np.sum(totals)) for i in range(len(vals))])


    def gain(self, X, y, insta ,splitter):
        """ calculates the gain for an attribute
        """
        tot_entrop = self.entropy(y[insta])

        vals, totals = np.unique(X[:, splitter], return_counts=True)

        """for i in range(len(vals)):
            print(f"y: {y}")
            print(f"X: {X}")

            print(f"weighted entropy: {[(totals[i] / np.sum(totals)) * self.entropy(y[np.where(X[:, splitter] == vals[i])[0], 0])]}")"""
        

        w_entrop = np.sum(                                 
            [(totals[i] / np.sum(totals)) * self.entropy(
            y[np.where(X[insta, splitter] == vals[i])[0], 0]) for i in range(len(vals))])
            # the ys (targets) where X in the split column == the respective value
        """
        entrops = []
        for i in range(len(vals)):
            split_array = np.where(X[:, splitter] == vals[i])[0]
            intersect = np.intersect1d(split_array, insta)
            split_y = y[intersect, 0]
            
            entrops.append((totals[i] / np.sum(totals)) * self.entropy(split_y))
        w_entrop = np.sum(np.array(entrops))"""


        return tot_entrop - w_entrop

    def id3(self, X, feat_set, y, insta, par_insta, par_class):
        """ ID3 algorithm
        """
        # only one out class? return it
        
        #print(f"uniq Ys {len(np.unique(y[insta, 0]))}")
        #print(f"the Ys {np.unique(y[insta, 0])}")
        if len(np.unique(y[insta, 0])) == 1:
            return np.unique(y[insta, 0])[0]

        # no data for branch? return parent node mode
        elif len(insta) == 0:
            return np.unique(y[par_insta, 0])[np.argmax(np.unique(y[par_insta, 0], return_counts=True)[1])]

        # no more features? return parent class
        elif len(feat_set) == 0:
            return par_class

        # further growth
        else:

            # grab parent class mode
            par_class = np.unique(y[insta, 0])[np.argmax(np.unique(y[insta, 0], return_counts=True)[1])]

            #print(f"X {X}")
            
            gain_vals = [self.gain(X, y, insta, feature) for feature in feat_set]
            most_gain = feat_set[np.argmax(gain_vals)]

            tree = {most_gain: {}}

            next_feat_set = [i for i in feat_set if i != most_gain]
            sub_insta = []
            # create a branch/leaf for each possible value of the most_gain column
            for val in np.unique(X[:, most_gain]):
                

                # make subsets split on different values for the most gain attribute
                branch_insta = np.where(X[:, most_gain] == val)[0].astype(np.int64)
                sub_insta = []
                for i in range(branch_insta.shape[0]):
                    if branch_insta[i] in insta:
                        sub_insta.append(branch_insta[i])
                # grow tree
                print(f"SPLIT {most_gain}")
                print(f"BRANCH {val}")
                print(f"NEXT_FEAT_set {next_feat_set}")
                #print(f"INSTA_set {insta}")
                print(f"BRANCH_INSTA {branch_insta}")
                print(f"SUB_INSTA {sub_insta}")
                print("")
                sub_tree = self.id3(X, next_feat_set, y, sub_insta, insta, par_class)

                # set branches/leafs
                tree[most_gain][val] = sub_tree
                

            return tree

    def eval(self, x: np.array , node: dict):
        """ Recursive evaluation for inference
        """
        # get the feature of the tree node
        # and get the instance of interest's
        # value for that feature
        node_name = [*node]
        x_feat_val = x[node_name[0]]


        try:
            branch = node[node_name[0]][x_feat_val]
        except:
            print(f"No branch from node: {node_name[0]}")
            print(f"for feature value: {x_feat_val}")
            return -1
            
        branch = node[node_name[0]][x_feat_val]
        #print(f"node name {node_name[0]}")
        #print(f"x feat val {x_feat_val}")
        #print(branch)

        if isinstance(branch, dict):
            return self.eval(x, branch)
        else:
            return branch
