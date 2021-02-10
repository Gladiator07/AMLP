from pandas.io import feather_format
import config
import time

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def run(fold):

    df = pd.read_csv(config.TRAINING_FILE)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(df[col])

        df.loc[:, col] = lbl.transform(df[col])


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_train[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )

    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")
    

if __name__ == "__main__":
    start_time = time.time()

    for fold_ in range(5):
        run(fold_)

    end_time = time.time() - start_time
    print(f"---{end_time} seconds---")

# git
