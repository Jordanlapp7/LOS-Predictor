import optuna
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def objective(trial, X, y, classify=True):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    if classify:
        objective_type = "multi:softmax"
        eval_metric = "merror"
        num_classes = len(np.unique(y))
    else:
        objective_type = "reg:squarederror"
        eval_metric = "rmse"
        num_classes = None

    param = {
        "objective": objective_type,
        "eval_metric": eval_metric,
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

    if classify:
        param["num_class"] = num_classes

    booster = xgb.train(
        param,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False
    )

    preds = booster.predict(dvalid)

    if classify:
        accuracy = accuracy_score(y_valid, preds)
        return 1 - accuracy  # Minimize error
    else:
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        return rmse

def run_optuna_tuning(X, y, n_trials=50, classify=False):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, classify), n_trials=n_trials)

    print("\nBest Score:", study.best_value)
    print("Best Parameters:", study.best_params)

    import optuna.visualization as vis
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    from_cache = True
    classify = True

    X, y = get_clean_data(from_cache, classify)
    run_optuna_tuning(X, y)