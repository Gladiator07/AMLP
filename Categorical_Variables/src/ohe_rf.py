
import config        # type: ignore
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble


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

    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df.valid.target.values, valid_preds)

    print(auc)


if __name__ == "__main__":

    for fold_ in range(5):
        run(fold_)

# Took forever to run
# Maybe not a good idea to one-hot encode for tree-based models in this problem 
# * Curse of dimensionality intesifies *