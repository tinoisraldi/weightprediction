import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_excel("weight-height.xlsx")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
X = df[["Height", "Gender"]]
y = df["Weight"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
import joblib

joblib.dump(regr, "clf.pkl")
