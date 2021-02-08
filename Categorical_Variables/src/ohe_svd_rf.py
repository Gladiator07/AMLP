# Running a RandomForest over a one-hot vector may take lot of time
# So, we reduce the sparse one-hot encoded matrices using singular value decomposition

import config        # type: ignore
import pandas as pd
import time

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv(config.TRAINING_FILE)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )

    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])

    x_valid = ohe.transform(df_valid[features])

    # we are reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit svd on full training data (train + valid)
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform sparse training data
    x_train = svd.transform(x_train)
    
    # transform sparse validation data
    x_valid = svd.transform(x_valid)

    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # as we are calculating AUC 
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    start_time = time.time()

    for fold_ in range(5):
        run(fold_)

    end_time = time.time() - start_time
    print(f"---{end_time} seconds---")

# This script too took forever like the previous ohe rf model