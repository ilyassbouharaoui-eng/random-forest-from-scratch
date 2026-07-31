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

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

