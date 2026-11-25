import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train = pd.read_csv("data/Training.csv").dropna(axis=1, how="all")
test = pd.read_csv("data/Testing.csv").dropna(axis=1, how="all")
for d in (train, test):
    d.columns = d.columns.str.strip()
    d["prognosis"] = d["prognosis"].str.strip()

X = train.drop(columns="prognosis")
le = LabelEncoder()
y = le.fit_transform(train["prognosis"])
X_test = test[X.columns]
y_test = le.transform(test["prognosis"])

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                    eval_metric="mlogloss").fit(X, y)

ens = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2
print("RandomForest :", accuracy_score(y_test, rf.predict(X_test)))
print("XGBoost      :", accuracy_score(y_test, xgb.predict(X_test)))
print("Ensemble     :", accuracy_score(y_test, ens.argmax(axis=1)))

joblib.dump({"rf": rf, "xgb": xgb, "le": le, "features": list(X.columns)},
            "model.joblib")
print("Saved model.joblib")