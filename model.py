"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    values,counts = np.unique(labels,return_counts=True)
    if (len(values)==0 )or (len(values) == 1) :
        return 0.0
    else:
        p = counts/len(labels)
        for i in p:
            if i ==1 :
                return 0.0
        return float(sum( -i*np.log(i) for i in p))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    col = features[:,feature_index]
    mask = col <=threshold
    lx = features[mask]
    ly = labels[mask]
    rx = features[~mask]
    ry = labels[~mask]
    return (lx,ly,rx,ry)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    xl = len(left_labels)/len(parent_labels)
    xr = len(right_labels)/len(parent_labels)
    return impurity(parent_labels) - (xl*impurity(left_labels) + xr*impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    dic = {'feature_index' : None,
    'threshold' : None,
    'score' : 0.0 }
    for i in feature_indices:
        x = features[:,i]
        l = [(x[i] + x[i+1])/2  for i in range(len(x)-1) ]
        for s in l : 
            lX, ly, rX, ry = split_dataset(features, labels, i, s )
            score = split_score(labels,ly,ry)
            if score > dic['score'] :
                dic['feature_index'] = i
                dic['threshold'] = s
                dic['score'] = score
    return dic

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if (len(np.unique(labels)) ==1) or ( depth >= max_depth) or (len(labels) < min_samples_split) :
        return True
    else:
        return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    #  choose a single class label to output for a leaf given the labels that reached it
    l,oc = np.unique(labels,return_counts = True)
    indice = np.argmax(oc)
    return int(l[indice])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    while not should_stop(labels, depth, max_depth, min_samples_split) :
        if feature_subset is not None:
            feature_indices = feature_subset
        else:
            feature_indices = range(len(features[0]))
        d = best_split(features, labels, feature_indices)
        if d['feature_index'] is None:
            return {
                'leaf': True,
                'prediction': leaf_prediction(labels)
            }
        res = {}
        res['leaf'] = False
        res['feature_index'] = d['feature_index']
        res['threshold'] = d['threshold']
        lx,ly,rx,ry = split_dataset(features, labels, res['feature_index'], res['threshold'] )
        depth += 1
        res['left'] =  build_tree(lx,ly,max_depth,min_samples_split,feature_subset,depth) 
        res['right'] = build_tree(rx,ry,max_depth,min_samples_split,feature_subset,depth) 
        return res
    return {'leaf' : True,
    'prediction' : leaf_prediction(labels)}

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    if tree['leaf']:
        return tree['prediction']
    else:
        if example[tree['feature_index']] <= tree['threshold']:
            return predict_example_tree(tree['left'],example)
        else:
            return predict_example_tree(tree['right'],example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    res = np.zeros(len(features))
    for i in range(len(features)) :
        res[i] = predict_example_tree(tree,features[i])
    return res

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # draw a bootstrap sample of rows (with replacement) using rng.
    x,y = np.zeros((len(features),len(features[0])),dtype = int) , np.zeros(len(labels),dtype = int)
    for i in range(len(labels)):
        j = rng.integers(len(labels))
        x[i] = features[j]
        y[i] = labels[j] 
    return x,y

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # return num_to_pick distinct random feature indices from range(num_features) using rng.
    x = rng.permutation(num_features)[:num_to_pick] 
    return   np.asarray(x, dtype=int)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    #grow num_trees decision trees on bootstrap samples with random feature subsets.
    n,d = len(features) , len(features[0])
    if  num_features_per_split == None :
        num_features_per_split = max(1,round(np.sqrt(d)))
    rng = np.random.default_rng(random_state)
    l = [{} for _ in range(num_trees)]
    for dic in l:
        x,y =  bootstrap_sample(features, labels, rng)
        x_i = feature_subset(d, num_features_per_split, rng)
        dic['tree'] = build_tree(x, y, max_depth, min_samples_split, feature_subset = x_i, depth=0)
        dic['feature_indices'] = x_i
    return l

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # aggregate the per-tree predictions of an ensemble into one prediction per example.
    num_trees, n = tree_predictions.shape
    res = np.zeros(n)
    for i in range(n) :
        l,c = np.unique(tree_predictions[:,i],return_counts=True)
        res[i] = l[np.argmax(c)]
    return res

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

