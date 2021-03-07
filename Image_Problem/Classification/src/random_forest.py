import os
from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


def create_dataset(training_df, image_dir):
    """
    This function takes the training dataframe and outputs
    training array and labels
    :param training_df: dataframe with ImageId, Target columns
    :param image_dir: location of images (folder), string
    :return: X, y (training array with features and labels)
    """

    # create empty list to store image vectors
    images = []

    # create empty list to store targets
    targets = []

    for index, row in tqdm(
        training_df.iterrows(),
        total=len(training_df),
        desc="processing images"
    ):
        # get image id
        image_id = row["ImageId"]

        # create image path
        image_path = os.path.join(image_dir, image_id)

        # open image using PIL
        image = Image.open(image_path + '.png')

        # resize image to 256x256. We use bilinear resampling
        image = image.resize((256, 256), resample=Image.BILINEAR)

        # convert image to numpy array
        image = np.array(image)

        # ravel (flatten)
        image = image.ravel()

        # append images and targets lists
        images.append(image)
        targets.append(int(row["target"]))
    
    # convert list of list of images to numpy array
    images = np.array(images)

    # print size of this array
    print(image.shape)
    return images, targets

if __name__ == "__main__":
    csv_path = "../input_image/train.csv"
    image_path = "../input_image/train_png/"

    # read CSV with imageid and target columns
    df = pd.read_csv(csv_path)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5, random_state=42)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    for fold in range(5):
        train_df = df[df.kfold != fold].reset_index(drop=True)
        test_df = df[df.kfold == fold].reset_index(drop=True)

        xtrain, ytrain = create_dataset(train_df, image_path)

        xtest, ytest = create_dataset(test_df, image_path)

        clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2)
        clf.fit(xtrain, ytrain)

        preds = clf.predict_proba(xtest)[:, 1]

        # print results
        print(f"FOLD: {fold}")
        print(f"AUC = {metrics.roc_auc_score(ytest, preds)}")
        print("")