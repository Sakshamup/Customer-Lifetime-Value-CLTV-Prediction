from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_save_model(X, y, model_path='models/cltv_model.pkl'):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingRegressor())
    ])

    param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__max_depth': [3, 5],
        'gb__learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 Score:", r2_score(y_test, y_pred))

    joblib.dump(best_model, model_path)
    return best_model
