import random
import numpy as np
import pandas as pd
import os
import richardsonpy.classes.occupancy as occ
import richardsonpy.functions.load_radiation as loadrad
import richardsonpy.classes.electric_load as eload
import OpenDHW

#===============================================================================================
# RICHARDSON ELECTRIC LOAD
def generate_hourly_el_load(csv_out='RICHARDSON.csv', nb_occ=3, seed=1, timestep=3600):

    random.seed(seed)
    np.random.seed(seed)

    occupancy_object = occ.Occupancy(number_occupants=nb_occ)
    occupancy_profile = occupancy_object.occupancy

    q_direct, q_diffuse = loadrad.get_rad_from_try_path()

    el_load_obj = eload.ElectricLoad(occ_profile=occupancy_profile,
                                     total_nb_occ=nb_occ,
                                     q_direct=q_direct,
                                     q_diffuse=q_diffuse,
                                     timestep=timestep)

    load = el_load_obj.loadcurve  # W

    if len(load) != 8760:
        raise RuntimeError(f'ERROR: EXPECTED 8760 VALUES')

    datetime = pd.date_range(start='2019-01-01', periods=8760, freq='h')

    df = pd.DataFrame({'datetime': datetime,'load_W': load})
    df.to_csv(csv_out, index=False)
    print(f'Wrote {len(df)} rows to {csv_out}')


def generate_multiple_el(n=10, base_seed=1, nb_occ=3, timestep=3600):
    
    for i in range(n):
        seed = base_seed + i
        csv_out = os.path.join( f'RICHARDSON_{seed}.csv')
        generate_hourly_el_load(csv_out=csv_out, nb_occ=nb_occ, seed=seed, timestep=timestep)


#===============================================================================================
# DHW

def generate_hourly_dhw_load(csv_out='DHW.csv', nb_occ=3, seed=1, mean_drawoff_vol_per_day=40, temp_dT=35):
 
    random.seed(seed)
    np.random.seed(seed)
    
    building_type = "SFH"           
    s_step = 60                      
    categories = 4                   
    
    holidays = OpenDHW.get_holidays(country_code="CH", year=2019)
    
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        categories=categories,
        occupancy=nb_occ,
        building_type=building_type,
        weekend_weekday_factor=1.2,  
        holidays=holidays,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        initial_day=1
    )
    
    timeseries_df = OpenDHW.compute_heat(timeseries_df=timeseries_df, temp_dT=temp_dT)
    
    timeseries_df["P_dhw_W"] = (timeseries_df["Heat_kWh"] * 1000.0 * 3600.0 / s_step)
    
    dhw_hourly = timeseries_df["P_dhw_W"].resample("h").mean()
    
    dhw_hourly = dhw_hourly.iloc[:8760]
    
    if len(dhw_hourly) != 8760:
        raise RuntimeError(f'ERROR: EXPECTED 8760 VALUES')
    
    df_out = dhw_hourly.reset_index()
    df_out.columns = ['datetime', 'dhw_W']
    
    df_out.to_csv(csv_out, index=False)
    print(f'Wrote {len(df_out)} rows to {csv_out}')

def generate_multiple_dhw(n=10, base_seed=1 , nb_occ=3, mean_drawoff_vol_per_day=40, temp_dT=35):
    
    for i in range(n):
        seed = base_seed + i
        csv_out = os.path.join( f'DHW_{seed}.csv')
        generate_hourly_dhw_load(csv_out=csv_out, nb_occ=nb_occ, seed=seed,
                                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day, temp_dT=temp_dT)


#===============================================================================================
# TEMPERATURES

def generate_temperature_data(csv_out='TEMPERATURES.csv', seed=1):

    np.random.seed(seed)
    
    ts = 8760
    
    # ambient: 13 to 25°C
    Tamb = np.random.uniform(13, 25, ts)
    # collector: Tamb + 0 to 10°C
    Tcoll = np.random.uniform(0, 10, ts) + Tamb
     
    datetime = pd.date_range(start='2019-01-01', periods=ts, freq='h')
    df = pd.DataFrame({'datetime': datetime, 'Tamb_C': Tamb, 'Tcoll_C': Tcoll,})
    
    df.to_csv(csv_out, index=False)
    print(f'Wrote {len(df)} rows to {csv_out}')
    
    
def generate_multiple_temps(n=10, base_seed=1):
    
    for i in range(n):
        seed = base_seed + i
        csv_out = os.path.join( f'TEMPERATURES_{seed}.csv')
        generate_temperature_data(csv_out=csv_out, seed=seed)



#===============================================================================================
# Seed

if __name__ == '__main__':
    
    n = 10
    base_seed = 1
    nb_occ = 3
    timestep = 3600
    mean_drawoff_vol_per_day = 40
    temp_dT = 35

    print(f"Generating CSVs")
    generate_multiple_el(n=n, base_seed=base_seed, nb_occ=nb_occ, timestep=timestep)
    generate_multiple_dhw(n=n, base_seed=base_seed,  nb_occ=nb_occ, mean_drawoff_vol_per_day=mean_drawoff_vol_per_day, temp_dT=temp_dT)
    generate_multiple_temps(n=n, base_seed=base_seed)
    