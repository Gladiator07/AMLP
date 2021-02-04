# src/train.py

# Training script

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    train_folds_path = '../input/mnist_train_folds.csv'
    df = pd.read_csv(train_folds_path)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf, f"../model/dt_{fold}.bin")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)