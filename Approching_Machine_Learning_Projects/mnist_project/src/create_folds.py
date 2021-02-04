from sklearn import model_selection
import pandas as pd

if __name__ == "__main__":
    train_path = "/mnt/d/Approaching(Almost)_Any_MachineLearning_Problem/Approching_Machine_Learning_Projects/mnist_project/input/mnist_train.csv"
    df = pd.read_csv(train_path)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    path = '/mnt/d/Approaching(Almost)_Any_MachineLearning_Problem/Approching_Machine_Learning_Projects/mnist_project/input/mnist_train_folds.csv'

    df.to_csv(path, index=False)
