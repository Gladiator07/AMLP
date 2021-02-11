# src/train.py

# Training script

import argparse
import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)

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

    # fetch the model from model dispatcher
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT, f"../model/dt_{fold}.bin")
                )


if __name__ == "__main__":

    # Bad way (may crash the program due to overloading memory)
    # run(fold=0)
    # run(fold=1)
    # run(fold=2)
    # run(fold=3)
    # run(fold=4)

    # Correct way to run folds

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add different arguments you need and their type
  
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the argument from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold,
        model=args.model)


# run the script by:
# python train.py --fold 0 --model decision_tree_gini