#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

np.random.seed(5)

# Parameters for wine quality dataset
# label_name is the column containing the class label, data is 
# a pandas dataframe containing the data in [feature1, feature2, ... featuren, label] format

#label_name = "quality"
#data = pd.read_csv("./data/wine/winequality-white.csv", delimiter=";")
#alpha = 0.00000001

# Parameters for UCI dataset

label_name = "label"
data = pd.DataFrame(np.loadtxt("./data/human/UCI HAR Dataset/train/X_train.txt"))
labels_data = np.genfromtxt("./data/human/UCI HAR Dataset/train/y_train.txt")
data[label_name] = labels_data
alpha = 0.0000001

print(data)

# Group into classes
groups = data.groupby(label_name)

# Calculate priors
labels = groups.groups.keys()
priors = groups.size() / len(data)

# Generate random variables
rvs = {}
for label, group in groups:
    features = group.drop(label_name, axis = 1)
    mean = features.mean()
    clean_cov = features.cov()

    # Calculate covariance with adjustment to make eigenvalues positive (C + (a * trace(C)/rank(C)) * I)
    cov = clean_cov + (np.eye(len(mean)) * alpha * (np.trace(clean_cov) / np.linalg.matrix_rank(clean_cov)))
    rvs[label] = multivariate_normal(mean.to_numpy(), cov.to_numpy())

def classify(features, priors, rvs):
    """
    Classify a feature vector, given class priors and random variables, using minimum error estimation
    """
    probs = {}
    for label in labels:
        chance = priors[label] * rvs[label].pdf(features)
        probs[label] = chance

    return max(probs, key=lambda k: probs[k])

# Calculate predicted class
data["predicted"] = data.apply(lambda row: classify(row.drop(label_name).to_numpy(), priors, rvs), axis = 1)

# Generate confusion matrix as a pandas crosstab
conf_matrix = pd.crosstab(data[label_name], data["predicted"], rownames=[label_name], colnames=["predicted"])

# Calculate error
matches = data[data[label_name] == data["predicted"]]
correct = len(matches) / len(data)
error = 1 - correct

print(conf_matrix)
print(f"Error rate of {error * 100}%")
