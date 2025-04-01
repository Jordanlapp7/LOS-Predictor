import xgboost as xgb
import itertools

def run_grid_search(X, y):
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
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            **combo
        }

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,
            nfold=5,
            early_stopping_rounds=10,
            seed=42,
            verbose_eval=False
        )

        score = cv_results['test-rmse-mean'].min()
        print(f"CV RMSE: {score:.4f}")

        if score < best_score:
            best_score = score
            best_params = combo

    print("\nBest Parameters:", best_params)
    print("Best RMSE:", best_score)

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    X, y = get_clean_data()
    run_grid_search(X, y)