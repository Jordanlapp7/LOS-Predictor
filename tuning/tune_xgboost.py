import xgboost as xgb
import itertools
import numpy as np

def run_grid_search(X, y, classify=True):
    if classify:
        # Convert string labels to integers
        classes, y = np.unique(y, return_inverse=True)
        num_class = len(classes)
        objective = 'multi:softmax'
        eval_metric = 'merror'
        score_key = 'test-merror-mean'
        better = min  # Lower error is better
    else:
        num_class = None
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        score_key = 'test-rmse-mean'
        better = min  # Lower RMSE is better

    dtrain = xgb.DMatrix(X, label=y)

    param_grid = {
        'max_depth': [4, 6],
        'eta': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }

    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = float("inf")
    best_params = None

    for combo in combos:
        print("Testing:", combo)
        params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'seed': 42,
            **combo
        }

        if classify:
            params['num_class'] = num_class

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,
            nfold=5,
            early_stopping_rounds=10,
            seed=42,
            verbose_eval=False
        )

        score = cv_results[score_key].min()
        print(f"CV {eval_metric.upper()}: {score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params

    print("\nBest Parameters:", best_params)
    print(f"Best {eval_metric.upper()}: {best_score:.4f}")

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data

    from_cache = True
    classify = True

    X, y = get_clean_data(from_cache, classify)
    run_grid_search(X, y, classify)