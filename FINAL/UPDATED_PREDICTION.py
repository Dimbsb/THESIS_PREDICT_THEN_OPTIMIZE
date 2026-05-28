# PREDICTION
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from UPDATED_EMS_BFS import solve_instance
from UPDATED_EMS_BB import load_data as ems_load_data
from sklearn.metrics import r2_score, mean_squared_error
########################################################################################################

# Load real data
def load_official_data(location: str, T: int, data_directory: str):
    data_directory = os.path.expanduser(data_directory)

    def _read(filename):
        path = os.path.join(data_directory, filename)
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.set_index("datetime").resample("h").mean().iloc[:T]
        return df

    electricity_df = _read(f"ELECTRICITY_{location}.csv")
    dhw_df = _read(f"DHW_{location}.csv")
    sph_df = _read(f"SPACE_HEAT_{location}.csv")
    temp_df = _read(f"TEMPERATURES_{location}.csv")

    df = pd.DataFrame(index=electricity_df.index)
    df["load_W"] = electricity_df["load_W"].values
    df["dhw_W"] = dhw_df["dhw_W"].values
    df["Q_space_W"] = sph_df["Q_space_W"].values
    df["Tamb_C"] = temp_df["Tamb_C"].values + 273.15

    return df

########################################################################################################

# Time, Day features for regression
def Time_Features(df):
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.dayofyear
    return df

########################################################################################################

# Building specs
def Building(df, building_name):
    building_specs = {"HOUSE1": {"U": 93, "C": 27 * 1e6}, "HOUSE2": {"U": 150, "C": 40 * 1e6}, "HOUSE3": {"U": 80, "C": 17 * 1e6}}

    base_building_name = building_name.split(".")[0]
    specs = building_specs.get(base_building_name, {"U": 93, "C": 27 * 1e6})
    df["U"] = specs["U"]
    df["C"] = specs["C"] / 1e6
    return df

########################################################################################################

# Combine all features
def Building_features(location: str, T: int, data_directory: str):
    house_name = os.path.basename(data_directory)
    house_df = load_official_data(location, T, data_directory)
    house_df = Time_Features(house_df)
    return Building(house_df, house_name)

########################################################################################################

# Regression model train
def train_and_predict(train_df, test_df):
    base_features = ["Tamb_C", "hour", "weekday", "day", "U", "C"]

    X_train = train_df[base_features]
    X_test = test_df[base_features]
    y_cols = ["load_W", "dhw_W", "Q_space_W"] #3

    r2_metric = {}
    y_prediction = {}

    for target_col in y_cols:
        y_train = train_df[target_col]
        y_test = test_df[target_col]
  
        #model = XGBRegressor(objective="reg:squarederror", n_estimators=500, learning_rate=0.05, random_state=42)
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        #model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_prediction[target_col] = y_pred_test

        r2_metric[target_col] = {"r2": r2_score(y_test, y_pred_test), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test))}

    return r2_metric, y_prediction

########################################################################################################

# MAIN
if __name__ == "__main__":
    locations = ["THESS", "ATH"]
    noise_levels = [0.0, 0.20]
    T = 8760
    data_directory = os.path.expanduser("~/DB_WORKSPACE")
    house_scenarios = {
        "HOUSE1": {"train": ["HOUSE1", "HOUSE1.2"], "test": "HOUSE1.1"},
        "HOUSE2": {"train": ["HOUSE2", "HOUSE2.2"], "test": "HOUSE2.1"},
        "HOUSE3": {"train": ["HOUSE3", "HOUSE3.2"], "test": "HOUSE3.1"}
        }

# Noise
    def add_noise(prediction_array, level):
        noise = np.random.normal(loc=0, scale=level * np.std(prediction_array), size=prediction_array.shape)
        return np.clip(prediction_array + noise, 0, None)

    for scenario_name, scenario in house_scenarios.items():
        train_houses = [os.path.join(data_directory, house_name) for house_name in scenario["train"]]
        test_data_directory = os.path.join(data_directory, scenario["test"])

        for location in locations:
            print("\n" + "=" * 60)
            print(f"----- Predict then Optimize: {scenario_name} -----")
            print("=" * 60)
            print(f"Train houses: {', '.join(scenario['train'])}")
            print(f"Test house:   {scenario['test']}")

            train_frames = []
            for house_direction in train_houses:
                train_frames.append(Building_features(location, T, house_direction))

            train_df = pd.concat(train_frames, axis=0).sort_index()
            test_df = Building_features(location, T, test_data_directory)

            r2_score_result, y_prediction = train_and_predict(train_df, test_df)

            electricity_score = f"[{location}] Forecast r2 (Electricity): {r2_score_result['load_W']['r2']:.4f}   RMSE: {r2_score_result['load_W']['rmse']:.2f} W"
            dhw_score = f"[{location}] Forecast r2 (Hot water):   {r2_score_result['dhw_W']['r2']:.4f}   RMSE: {r2_score_result['dhw_W']['rmse']:.2f} W"
            sph_score = f"[{location}] Forecast r2 (Space Heat):  {r2_score_result['Q_space_W']['r2']:.4f}   RMSE: {r2_score_result['Q_space_W']['rmse']:.2f} W"
            print(electricity_score)
            print(dhw_score)
            print(sph_score)

########################################################################################################

# PERFECT INFORMATION
            print(f"\nPerfect Information {location}")
            print("-" * 40)
            perfect_data = ems_load_data(location, T, test_data_directory)
            perfect_result = solve_instance(inputs=perfect_data)
            perfect_cost = perfect_result["bb_cost"]

            if not np.isfinite(perfect_cost):
                perfect_cost = None

########################################################################################################

# FORECASTED INFORMATION
            for noise in noise_levels:
                forecast_line = f"Forecasted Demand with Noise={noise*100:.0f}% for {location}"
                print(f"\n{forecast_line}")
                print("-" * 40)
                predicted_data = ems_load_data(location, T, test_data_directory)
                predicted_data["L_electricity"] = add_noise(y_prediction["load_W"], noise)
                predicted_data["L_dhw"] = add_noise(y_prediction["dhw_W"], noise)
                predicted_data["L_sph"] = add_noise(y_prediction["Q_space_W"], noise)

                prediction_result = solve_instance(inputs=predicted_data)
                prediction_cost = prediction_result["bb_cost"]

                if not np.isfinite(prediction_cost):
                    continue

########################################################################################################

# ΑΠΟΤΕΛΕΣΜΑΤΑ
                if perfect_cost is not None and prediction_cost is not None:
                    cost_gap = prediction_cost - perfect_cost

                    print("\n" + "-" * 40)
                    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ [{location}]")
                    print(f"Noise Level: {noise*100:.0f}%")
                    print("-" * 40)
                    print(f"Perfect information:            {perfect_cost:,.2f} CHF")
                    print(f"Forecasted information:         {prediction_cost:,.2f} CHF")
                    print(f"Forecast - Perfect Info:        {cost_gap:,.2f} CHF")
                    print("-" * 40 + "\n")
