import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

def run_training(fold):
    df = pd.read_csv("../input_text/train_folds.csv")
    df.review = df.review.apply(str)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)
    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    clf = linear_model.LogisticRegression()
    clf.fit(xtrain, ytrain)

    pred = clf.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)

    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "lr_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "lr_pred"]]

if __name__=="__main__":
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/lr.csv", index=False)