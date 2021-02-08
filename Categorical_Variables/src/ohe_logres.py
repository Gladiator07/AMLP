# one-hot encoding + Logistic Regression

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import config        # type: ignore
import time

def run(fold):
    # loading the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # all columns are features except id, target and kfold column

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # fill all NaN values with NONE

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # initialize the OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )

    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # intialize Logistic Regression model
    model = linear_model.LogisticRegression()

    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need probability as we are calculating AUC
    # we will use probability of ones

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(auc)


if __name__ == "__main__":
    # run for fold 0
    # run(0)

    # run for all folds
    start_time = time.time()

    for fold_ in range(5):
        run(fold_)

    end_time = time.time() - start_time
    print(f"--- {end_time} seconds ---")



# This script produces following results:
# 0.7861181966678912
# 0.7850866764666862
# 0.7880526183419283
# 0.7859593431685528
# 0.786313353749265
# --- 60.28842424245 seconds ---