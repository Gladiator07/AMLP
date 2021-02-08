import config
import pandas as pd
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
    for fold_ in range(5):
        run(fold_)

# With OHE
# Fold = 0, AUC = 0.7175068247592911
# Fold = 1, AUC = 0.7153852119482599
# Fold = 2, AUC = 0.717459784306343
# Fold = 3, AUC = 0.7154430139392741
# Fold = 4, AUC = 0.7163421012124247