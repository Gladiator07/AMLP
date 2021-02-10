# This script can be used with any dataset for KFold cross validation
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    train_path = 'cross-validation/winequality-red.csv'
    df = pd.read_csv(train_path)

    # creating a new column 'kfold' and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # intiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, [trn_, val_] in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)

 # Results 