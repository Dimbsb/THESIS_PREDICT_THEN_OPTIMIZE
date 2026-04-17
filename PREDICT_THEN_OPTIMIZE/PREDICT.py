import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from UPDATED_EMS_BB import load_data as ems_load_data
from UPDATED_EMS_BFS import solve_instance

# Load real data
def load_and_prepare_data(location: str, T: int, data_directory: str):
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
    df['load_W'] = electricity_df['load_W'].values
    df['dhw_W'] = dhw_df['dhw_W'].values
    df['Q_space_W'] = sph_df['Q_space_W'].values
    df['Tamb_C'] = temp_df['Tamb_C'].values + 273.15 
    
    return df

##############################################################################################

# Temperature, Time, Day features for regression
def addTimeFeatures(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day'] = df.index.dayofyear
    return df

##############################################################################################

# Regression model train
def train_and_predict(df):
    X = df[['Tamb_C', 'hour', 'weekday', 'day']]
    y_cols = ['load_W', 'dhw_W', 'Q_space_W']

    rmse_score = {}
    y_prediction = {}

    for target_col in y_cols:
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_prediction[target_col] = model.predict(X)   
        rmse_score[target_col] = rmse(y_test, model.predict(X_test))

    return rmse_score, y_prediction

##############################################################################################

# ROOT MEAN SQUARED ERROR 
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred)))

##############################################################################################
# MAIN 
if __name__ == "__main__":
    
    locations = ['THESS', 'ATH']
    noise_levels = [0.0, 0.20]   
    T = 8760
    data_directory = "~/DB_WORKSPACE/HOUSE3"

    def add_noise(prediction_array, level):
        noise = np.random.normal(loc=0, scale=level * np.std(prediction_array), size=prediction_array.shape)
        return np.clip(prediction_array + noise, 0, None)

    for location in locations:
        print(f"\n" + "="*60)
        print(f"--------- Predict then Optimize ---------")
        print("="*60)
        
        df = load_and_prepare_data(location, T, data_directory)
        df = addTimeFeatures(df.dropna())
        
        rmse_score, y_prediction = train_and_predict(df)

        print(f"[{location}] Forecast RMSE (Electricity):        {rmse_score['load_W']:.2f} W")
        print(f"[{location}] Forecast RMSE (Hot water):          {rmse_score['dhw_W']:.2f} W")
        print(f"[{location}] Forecast RMSE (Space Heat):         {rmse_score['Q_space_W']:.2f} W")

############################################################################################## 

# PERFECT INFORMATION 
        print(f"\nPerfect Information {location}")

        perfect_data = ems_load_data(location, T, data_directory)
        perfect_result = solve_instance(inputs=perfect_data)
        perfect_cost = perfect_result["bb_cost"]

        if not np.isfinite(perfect_cost): perfect_cost = None

##############################################################################################

# FORECASTED INFORMATION 
        for noise in noise_levels:
            print(f"\nForecasted Demand with Noise={noise*100:.0f}% for {location}")
            
            predicted_data = ems_load_data(location, T, data_directory)
            predicted_data['L_electricity'] = add_noise(y_prediction['load_W'],    noise)
            predicted_data['L_dhw']         = add_noise(y_prediction['dhw_W'],     noise)
            predicted_data['L_sph']         = add_noise(y_prediction['Q_space_W'], noise)

            prediction_result = solve_instance(inputs=predicted_data)
            prediction_cost = prediction_result["bb_cost"]

            if not np.isfinite(prediction_cost): continue

##############################################################################################

# ΑΠΟΤΕΛΕΣΜΑΤΑ
            if perfect_cost is not None and prediction_cost is not None:
                print(f"\n" + "-"*40)
                print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ [{location}]")
                print(f"Noise Level: {noise*100:.0f}%")
                print(f"-"*40)
                print(f"Perfect information:            {perfect_cost:,.2f} CHF")
                print(f"Forecasted information:         {prediction_cost:,.2f} CHF")
                print(f"Forecast - Perfect Info:        {prediction_cost - perfect_cost:,.2f} CHF")
                print("-"*40 + "\n")