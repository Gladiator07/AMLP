import config # type: ignore
import pandas as pd
import time
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
    
    for col in features:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(df[col])

        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    
    x_valid = df_valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs=-1)

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



# Output:

# Fold = 0, AUC = 0.7160334115968606
# Fold = 1, AUC = 0.7146399857024811
# Fold = 2, AUC = 0.716906795007507
# Fold = 3, AUC = 0.7140732474138755
# Fold = 4, AUC = 0.7148884683614005
# ---218.38796615600586 seconds---