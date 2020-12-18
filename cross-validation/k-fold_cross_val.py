from sklearn import model_selection
import pandas as pd

if __name__ == "main":
    train_path = 'cross-validation/winequality-red.csv'
    df = pd.read_csv(train_path)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.quality

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv("train_stratified_folds.csv", index=False)