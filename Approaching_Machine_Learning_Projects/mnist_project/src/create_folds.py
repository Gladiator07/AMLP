from sklearn import model_selection
import pandas as pd
import config

if __name__ == "__main__":
    train_path = config.TRAINING_FILE_ORIGINAL
    df = pd.read_csv(train_path)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    path = '../input/mnist_train_folds.csv'

    df.to_csv(path, index=False)
