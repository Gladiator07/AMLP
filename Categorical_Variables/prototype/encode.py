# Encoding for rare variables which will occur in test data and not in training data

import pandas as pd
from sklearn import preprocessing

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# create a fake target column for test data 
# since this column doesn't exist
test.loc[:, "target"] = -1

# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)

# make a list of features we are interested in
# id and target is something we should not encode

features = [x for x in train.columns if x not in ["id", "target"]]

# loop over the features list
for feat in features:
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEncoder()

    temp_col = data[feat].fillna("NONE").astype(str).values

    # we can use fit_transform here as we do not 
    # have any extra test data we need to 
    # transform on separately

    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)

# split the training and test data again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)

