import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.sklearn.autolog()

df = pd.read_csv("dataset_preprocessing/train_clean.csv")

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid = {
    "n_estimators": [300],
    "max_depth": [None],
}

with mlflow.start_run(run_name="random_forest_autolog"):
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", acc)
    print("F1-score:", f1)
