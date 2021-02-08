# One-Hot Encoded Logistic Regression model cross-validated using sklearn

import config        # type: ignore
import pandas as pd
import time
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection

start_time = time.time()

df = pd.read_csv(config.TRAINING_FILE_ORIG)

features = [
    f for f in df.columns if f not in ("id", "target")
]

for col in features:
    df.loc[:, col] = df[col].astype(str).fillna("NONE")


ohe = preprocessing.OneHotEncoder()

X = ohe.fit_transform(df[features])

y = df.target.values

model = linear_model.LogisticRegression()

cv = model_selection.StratifiedKFold(n_splits=5)

scores = model_selection.cross_val_score(model, X, y, cv=cv)

print(scores)

end_time = time.time() - start_time
print(f"--- {end_time} seconds ---")

# 30 seconds to run