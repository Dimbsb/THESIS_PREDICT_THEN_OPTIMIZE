#########################################################
#########################################################
#########################################################
!pip install git+https://github.com/NREL/OCHRE.git

!pip install "numpy<2.0"

!pip uninstall gurobipy -y
!pip install gurobipy==12.0.0

print("\nRestart Session")

#########################################################
#########################################################
#########################################################

df, metrics, hourly = dwelling.simulate()

print("Διαθέσιμες στήλες:")
print(df.columns.tolist())
#########################################################
#########################################################
#########################################################

import os
import datetime as dt
from ochre import Dwelling
from ochre.utils import default_input_path
from pvlib.iotools import read_epw
import pandas as pd
import numpy as np

dwelling = Dwelling(
    start_time=dt.datetime(2018, 1, 1, 0, 0),
    time_res=dt.timedelta(minutes=60),
    duration=dt.timedelta(days=365),
    hpxml_file=os.path.join(default_input_path, 'Input Files', 'bldg0112631-up11.xml'),
    weather_file=os.path.join(default_input_path, 'Weather', 'USA_CO_Denver.Intl.AP.725650_TMY3.epw'),
)

df, metrics, hourly = dwelling.simulate()


# EXPORTS 

# 1. Ηλεκτρισμός 
pd.DataFrame({
    'datetime': df.index,
    'load_W': np.minimum(df['Total Electric Power (kW)'].values * 1000, 8000)
}).to_csv('RICHARDSON_OK.csv', index=False)

# 2. Ζεστό Νερό 
pd.DataFrame({
    'datetime': df.index,
    'dhw_W': np.minimum(df['Water Heating Electric Power (kW)'].values * 1000, 8000)
}).to_csv('DHW_OK.csv', index=False)

# 3. Θέρμανση Χώρου 
pd.DataFrame({
    'datetime': df.index,
    'Q_space_W': np.minimum(df['HVAC Heating Electric Power (kW)'].values * 1000, 8000)
}).to_csv('SPACE_HEAT_OK.csv', index=False)

# Ανάγνωση καιρού
weather_file = os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw")
weather_df, _ = read_epw(weather_file)
weather_df = weather_df.iloc[:len(df)]

# 4. Solar Data (GHI)
pd.DataFrame({
    'datetime': df.index,
    'GHI': weather_df['ghi'].values
}).to_csv('SOLAR_DATA_OK.csv', index=False)

# 5. Temperatures (Ambient & Collector)
pd.DataFrame({
    'datetime': df.index,
    'Tamb_C': weather_df['temp_air'].values + 7,
    'Tcoll_C': weather_df['temp_air'].values + 20
}).to_csv('TEMPERATURES_OK.csv', index=False)

print("OK")