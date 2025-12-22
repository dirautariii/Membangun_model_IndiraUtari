import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

dagshub.init(
    repo_owner="dirautariii",
    repo_name="Membangun_model",
    mlflow=True
)

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
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
}

with mlflow.start_run(run_name="random_forest_tuning"):
    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("max_depth", grid.best_params_["max_depth"])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(best_model, "model")

    disp = ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test
    )
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    print("Best params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1-score:", f1)
