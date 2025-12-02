import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def load_data(path: str) -> pd.DataFrame:
    """
    Load the F1 dataset from a CSV file.
    """
    df = pd.read_csv(path)
    print("Initial dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning rules described in the prompt.
    """

    for col in ['finish_position', 'grid_start_position']:
        if col in df.columns:
            df[col] = df[col].replace('/N', 20)  # "/N" -> 20
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(20).astype(int)

    mean_fill_cols = ['temperature', 'precipitation', 'windspeed']
    for col in mean_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)

    zero_fill_cols = [
        'driver_standing_podiums',
        'driver_standing_dnf_rate_n',
        'driver_standing_points_n',
        'driver_standing_laps_led',
        'driver_standing_points',
        'driver_standing_position',
        'driver_standing_wins'
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

    print("\nAfter preprocessing:")
    print(df[mean_fill_cols + zero_fill_cols[:3] + ['finish_position', 'grid_start_position']].head())
    print("\nMissing values after preprocessing:")
    print(df.isna().sum())

    return df


def train_and_evaluate_models(df: pd.DataFrame):
    """
    Train Linear Regression, Random Forest, XGB, CatBoost, LightGBM
    and print evaluation metrics.
    """

    target_col = 'finish_position'
    assert target_col in df.columns, f"{target_col} must be in dataframe."

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_features = ['driverId', 'constructorId', 'circuitId']
    cat_features = [c for c in cat_features if c in X.columns]

    X = pd.get_dummies(X, columns=cat_features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    def evaluate_model(model, name: str):
        """
        Fit model, predict on train & test, and print detailed metrics.
        """

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\n================ {name} ================")
        print("TRAIN METRICS:")
        print(f"  MAE :  {train_mae:.4f}")
        print(f"  RMSE:  {train_rmse:.4f}")
        print(f"  R²   :  {train_r2:.4f}")

        print("\nTEST METRICS:")
        print(f"  MAE :  {test_mae:.4f}")
        print(f"  RMSE:  {test_rmse:.4f}")
        print(f"  R²   :  {test_r2:.4f}")

        # Interpretation Helper
        print("\nInterpretation:")
        if train_rmse < test_rmse * 0.6:
            print("⚠️ Overfitting likely — Train error much lower than Test.")
        elif train_rmse > test_rmse * 1.5:
            print("⚠️ Underfitting — Model performs worse on Train.")
        else:
            print("✅ Good balance — Train and Test errors are similar.")
        print("============================================\n")



    lin_reg = LinearRegression()
    evaluate_model(lin_reg, "Linear Regression")

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    evaluate_model(rf, "Random Forest")

    # 4.c XGBoost
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    evaluate_model(xgb_model, "XGBoost")

        # 4.c XGBoost - more regularization, shallower trees, larger min_child_weight
    xgb_model_reg = XGBRegressor(
        n_estimators=400,          
        learning_rate=0.03,        
        max_depth=4,              
        subsample=0.7,             
        colsample_bytree=0.7,      
        min_child_weight=5,        
        reg_lambda=10.0,           
        reg_alpha=1.0,            
        gamma=1.0,                 
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    evaluate_model(xgb_model, "XGBoost (Regularized)")


    # 4.e LightGBM - moderately regularized (less extreme than before)
    lgbm_model = LGBMRegressor(
        n_estimators=400,          
        learning_rate=0.05,        
        max_depth=5,               
        num_leaves=31,             
        min_data_in_leaf=30,       
        subsample=0.8,             
        colsample_bytree=0.8,      
        reg_lambda=2.0,            
        reg_alpha=0.0,             
        min_gain_to_split=0.0,     
        random_state=42
    )
    evaluate_model(lgbm_model, "LightGBM (Moderately Regularized)")


    # 4.d CatBoost (silent)
    cat_model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        random_state=42,
        verbose=0
    )
    evaluate_model(cat_model, "CatBoost")

    return xgb_model


def predict_new_csv(model, csv_path: str, output_path: str = "xgb_predictions.csv"):
    """
    Predict finish_position for a new CSV using the trained XGBoost model.
    Applies preprocessing + column removal + one-hot alignment.
    """

    print(f"\nLoading file for prediction: {csv_path}")
    new_df = pd.read_csv(csv_path)

    output_df = new_df.copy()

    
    drop_cols = ['resultId', 'raceId', 'driverId', 'constructorId', 'circuitId']
    new_df = new_df.drop(columns=[c for c in drop_cols if c in new_df.columns], errors='ignore')

    new_df = preprocess(new_df)

    if "finish_position" in new_df.columns:
        new_df = new_df.drop(columns=["finish_position"])

    cat_cols = new_df.select_dtypes(include=['object']).columns.tolist()

    if len(cat_cols) > 0:
        new_df = pd.get_dummies(new_df, columns=cat_cols, drop_first=True)

    required_cols = model.feature_names_in_

    missing_cols = set(required_cols) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = 0

    new_df = new_df[required_cols]

    preds = model.predict(new_df)

    
    output_df["predicted_finish_position"] = preds

    
    output_df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(output_df.head())

    return output_df


def main():
    csv_path = "df_20_25.csv"
    df = load_data(csv_path)
    df = df.drop(['resultId', 'raceId', 'driverId', 'constructorId', 'circuitId'], axis = 1)

    df = preprocess(df)

    # Train models + return best model (CatBoost)
    best_model = train_and_evaluate_models(df)

    # Predict on new data file
    dfp = predict_new_csv(best_model, "df_pred.csv", "predictions_output.csv")

    return dfp

if __name__ == "__main__":
    df = main()
