# Running a RandomForest over a one-hot vector may take lot of time
# So, we reduce the sparse one-hot encoded matrices using singular value decomposition

import config        # type: ignore
import pandas as pd

from scipy import sparse
from sklearn import decomposition
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

    df_train = df[df.kfold != fold].reset_index(drop=True)