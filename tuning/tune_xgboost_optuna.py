import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def objective(trial, X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
        "seed": 42
    }

    booster = xgb.train(
        param,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False
    )

    preds = booster.predict(dvalid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse


def run_optuna_tuning(X, y, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Best RMSE:", study.best_value)
    print("Best Params:", study.best_params)

    # import optuna.visualization as vis
    # vis.plot_optimization_history(study).show()
    # vis.plot_param_importances(study).show()

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    X, y = get_clean_data(True)
    run_optuna_tuning(X, y)