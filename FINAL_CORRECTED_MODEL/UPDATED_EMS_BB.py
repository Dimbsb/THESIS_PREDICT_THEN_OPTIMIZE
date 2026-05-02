# EMS_model.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import math
from math import pi

# DATA LOADER
def load_data(location: str, T: int, data_directory: str) -> dict:
    import os
    data_directory = os.path.expanduser(data_directory)    

    def _read(filename, col):
        path = os.path.join(data_directory, filename)
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.set_index("datetime").resample("h").mean().iloc[:T]
        return df[col].values

    I_t           = _read(f"SOLAR_DATA_{location}.csv",  "GHI")
    L_electricity = _read(f"ELECTRICITY_{location}.csv", "load_W")
    L_dhw         = _read(f"DHW_{location}.csv",         "dhw_W")
    L_sph         = _read(f"SPACE_HEAT_{location}.csv",  "Q_space_W")

    temp_df = pd.read_csv(os.path.join(data_directory, f"TEMPERATURES_{location}.csv"), parse_dates=["datetime"])    
    temp_df = temp_df.set_index("datetime").resample("h").mean().iloc[:T]
    Tamb  = temp_df["Tamb_C"].values  + 273.15
    Tcoll = temp_df["Tcoll_C"].values + 273.15

    return dict(I_t=I_t, Tamb=Tamb, Tcoll=Tcoll, L_electricity=L_electricity, L_dhw=L_dhw, L_sph=L_sph, location=location, T=T)

# MODEL
def create_ems_model(T: int, I_t: np.ndarray, Tamb: np.ndarray, Tcoll: np.ndarray, L_electricity: np.ndarray, L_dhw: np.ndarray, L_sph: np.ndarray,) -> tuple:


    model = gp.Model("EMS")
    model.Params.OutputFlag = 0       
    
    # Decision Variables
    # Fuel Cell
    x_gas_fc = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_gas_fc") 
    x_gas_in_fc = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_gas_in_fc")
    x_sph_out_fc = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_fc")    
    x_dhw_out_fc = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_fc")  
    x_el_out_fc = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_el_out_fc")
    
    # PV
    x_el_pv = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_el_pv")
    x_el_out_pv = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_el_out_pv")
    
    # Solar Thermal
    x_th_st = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_th_st")
    x_sph_out_st = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_st")     
    x_dhw_out_st = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_st") 

    # Heat Pump
    x_el_hp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_el_hp")
    x_sph_out_hp = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_hp")
    x_el_in_hp = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_el_in_hp") 
    
    # Boiler
    x_gas_boiler = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_gas_boiler")
    x_gas_in_boiler = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_gas_in_boiler")  
    x_sph_out_boiler = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_boiler")  
    x_dhw_out_boiler = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_boiler")
    
    # Battery
    y_el_battery = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_el_battery")
    y_el_in_battery = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="y_el_in_battery")   
    y_el_out_battery = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="y_el_out_battery")     
    
    # Heat Tank 
    y_h_tank = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_h_tank") 
    y_dhw_net_tank = model.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_dhw_net_tank")
    y_dhw_in_tank = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="y_dhw_in_tank")   
    y_dhw_out_tank = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="y_dhw_out_tank") 
        
    # 3.13
    T_bdg = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="T_bdg") 

    # 3.14
    dT = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="dT") 

    print("DECISION VARIABLES OK")

    # Parameters - Table 4 - Table 5
    # Conversion Technologies

    # Fuel Cell Parameters
    fc_Cinv = 4000  
    fc_Comex = 200  
    fc_Cfuel = 0.1318 
    fc_Tlife = 15  
    fc_Ccap = 535  
    fc_h_el = 0.50  
    fc_h_th = 0.40  
    fc_pco2 = 200  
    fc_k = 0.30  

    # PV Parameters
    pv_Cinv = 2500 
    pv_Comex = 36  
    pv_Tlife = 25  
    pv_Ccap = 180  
    pv_h_el = 0.16  
    pv_h_pr = 0.90  
    pv_p = 144  

    # Solar Thermal Parameters
    st_Cinv = 770 
    st_Comex = 28  
    st_Tlife = 20  
    st_Ccap = 80  
    st_p = 650  
    st_h = 0.80  
    st_K = 4.0  

    # Heat Pump Parameters
    hp_Cinv = 1500 
    hp_Comex = 99  
    hp_Tlife = 23  
    hp_Ccap = 190  
    hp_h_cop = 4  
    hp_Cfuel = 0.0555  
    hp_pco2 = 91.5  

    # Boiler Parameters
    boiler_Cinv = 178 
    boiler_Comex = 14  
    boiler_Cfuel = 0.1186  
    boiler_Tlife = 17  
    boiler_Ccap = 28  
    boiler_h = 0.95 
    boiler_pco2 = 179  

    # Storage Technologies
    # Battery Parameters
    battery_Cinv = 1000  
    battery_Tlife = 9  
    battery_Ccap = 128  
    battery_h = 0.92  
    battery_E = 1.0 - (0.05 / 24.0)
    battery_mo = 0.20  
    battery_p_plus = 0.58  
    battery_p_minus = 1.00  
    battery_yo  = 1.00 
    battery_yT = 1.00 
    # 3.10
    battery_h_plus = np.sqrt(battery_h)
    battery_h_minus = 1.0/np.sqrt(battery_h)

    # Heat Store Parameters
    heat_store_Cinv = 86 
    heat_store_Tlife = 20 
    heat_store_Ccap = 6 
    heat_store_F = 0.80 
    heat_store_U = 0.21 
    heat_store_Cp = 4.19e3   
    heat_store_p = 1000  
    heat_store_Thot = 60+273.15 
    heat_store_Tcold = 10+273.15 
    heat_store_Tbdg = 23+273.15 
    heat_store_yo = 0.50  
    heat_store_yT = 0.50  
    heat_store_mo = 0.50  

    # Electrical grid
    el_grid_in = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="el_grid_in")
    el_grid_cost = 0.15
    el_grid_connection_fee = 130.0
    grid_pco2 = 250.0
    
    
    # 3.13
    U = 93
    C = 27 * 1e6

    # 3.14 
    c_T = 0.1
    T_min = 19+273.15 
    T_max = 27+273.15

    # SUM
    SOC_el = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="SOC_el")
    SOC_thermal = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="SOC_thermal")

    Dt = 3600
    co2_price = 0.06

    print("PARAMETERS OK")

    # lower bounds W
    min_cap_fc = 1000      
    min_cap_pv = 1000     
    min_cap_st = 1000       
    min_cap_hp = 1000       
    min_cap_boiler = 1000   
    min_cap_battery = 1 * 3.6e6  
    min_height_tank = 0.5    

    # Big M  
    Big_M = 40000.0   
    Big_M_Joules = 50.0 * 3.6e6  
    Big_M_meters = 5.0  

 
    # Binary Decision Variables 1 or 0  MADE CONTINUOUS   
    binary_fc = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_fc")
    binary_pv = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_pv")
    binary_st = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_st")
    binary_hp = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_hp")
    binary_boiler = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_boiler")
    binary_battery = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_battery")
    binary_tank = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="binary_tank")
    
    binary_vars = [binary_fc, binary_pv, binary_st, binary_hp, binary_boiler, binary_battery, binary_tank]

    print("BINARY VARIABLES OK")

    for t in range(T):
        # 3.1
        model.addConstr(fc_k * x_gas_fc <= x_gas_in_fc[t], name=f"fc_min_load{t}")
        # 3.1
        model.addConstr(x_gas_in_fc[t] <= x_gas_fc, name=f"fc_max_load{t}")
        # 3.1
        model.addConstr(x_el_out_fc[t] <= fc_h_el * x_gas_in_fc[t], name=f"fc_electricity_ub{t}")
        # 3.1
        model.addConstr(x_sph_out_fc[t] + x_dhw_out_fc[t] <= fc_h_th * x_gas_in_fc[t], name=f"fc_heat_ub{t}")
    
        # 3.3  
        model.addConstr(x_el_out_pv[t] <= (pv_h_el * pv_h_pr * (I_t[t]) * (x_el_pv / pv_p)), name=f"pv_electricity_ub{t}")
        
        # 3.4  
        #model.addConstr((x_sph_out_st[t] + x_dhw_out_st[t]) <= (((st_h * I_t[t]) - st_K * (Tcoll[t] - Tamb[t])) * x_th_st / st_p), name=f"st_heat_ub{t}")
        st_constraint = max(0.0, st_h * I_t[t] - st_K * (Tcoll[t] - Tamb[t]))
        model.addConstr((x_sph_out_st[t] + x_dhw_out_st[t]) <= st_constraint * x_th_st / st_p, name=f"st_heat_ub{t}")
        
        # 3.6 
        model.addConstr(x_sph_out_hp[t] <= hp_h_cop * x_el_hp, name=f"hp_space_heat_ub{t}")
        
        # 3.7 
        model.addConstr(x_el_in_hp[t] == x_sph_out_hp[t] / hp_h_cop, name=f"hp_electricity_consumption_rule{t}")
        
        # 3.8 
        model.addConstr(x_gas_in_boiler[t] <= x_gas_boiler, name=f"boiler_max_load{t}")
        
        # 3.8  
        model.addConstr((x_sph_out_boiler[t] + x_dhw_out_boiler[t]) <= boiler_h * x_gas_in_boiler[t], name=f"boiler_heat_ub{t}")

    print("CONVERSION CONSTRAINTS OK")
      
      
    for t in range(T):
        # 3.10  
        model.addConstr(y_el_in_battery[t] <= battery_p_plus * (y_el_battery /Dt), f"battery_charge_ub{t}")

        # 3.10
        model.addConstr(y_el_out_battery[t] <= battery_p_minus * (y_el_battery /Dt), f"battery_discharge_ub{t}")
 

    # 3.10
    model.addConstr(SOC_el[0] == battery_E * (battery_yo * y_el_battery) + (battery_h_plus * y_el_in_battery[0] - battery_h_minus * y_el_out_battery[0]) * Dt, name="battery_soc_dynamics_zero")

    # 3.10
    for t in range(1, T):
        model.addConstr(SOC_el[t] == (battery_E * SOC_el[t-1] + (battery_h_plus * y_el_in_battery[t] - battery_h_minus * y_el_out_battery[t]) * Dt), name=f"battery_soc_dynamics{t}")

    # 3.10 
    model.addConstrs((battery_mo * y_el_battery <= SOC_el[t] for t in range(T)), name="battery_soc_lower")

    # 3.10  
    model.addConstrs((SOC_el[t] <= y_el_battery for t in range(T)), name="battery_soc_upper")

    # 3.10  
    model.addConstr(SOC_el[T-1] == battery_yT * y_el_battery, "battery_soc_terminal")
    
    # 3.10 
    model.addConstr(sum(y_el_out_battery[t] for t in range(T)) * Dt <= (T / 24.0) * battery_h_minus * (1 - battery_mo) * y_el_battery, name="battery_cycling_limit")
 
    # 3.11  
    heat_store_capacity = (heat_store_p * heat_store_Cp * (heat_store_Thot - heat_store_Tcold) * pi * ((heat_store_F / 2.0)**2) * y_h_tank)

    # M1 3.11 
    heat_store_m1 = (pi * heat_store_U * (heat_store_Thot - heat_store_Tbdg) * (heat_store_F**2) / 2.0)

    # M2 3.11  
    heat_store_m2 = pi * heat_store_U * ((heat_store_Thot - heat_store_Tcold)/2.0 - (heat_store_Tbdg - 273.15)) * heat_store_F * y_h_tank

    # 3.12  
    model.addConstr(SOC_thermal[0] == heat_store_yo * heat_store_capacity + y_dhw_net_tank[0] * Dt - (heat_store_m1 + heat_store_m2) * Dt, name="tank_soc_dynamics_t0")

    # 3.12  
    for t in range(1, T):
        model.addConstr(SOC_thermal[t] == SOC_thermal[t-1] + y_dhw_net_tank[t] * Dt - (heat_store_m1 + heat_store_m2) * Dt, name=f"tank_soc_dynamics{t}")

    # 3.12  
    model.addConstrs((SOC_thermal[t] <= heat_store_capacity for t in range(T)), name="tank_soc_upper")

    # 3.12  
    model.addConstr(SOC_thermal[T-1] == heat_store_yT * heat_store_capacity, name="tank_soc_terminal")


    print("STORAGE CONSTRAINTS OK")
    
 
    # 3.13  
    tau = C / U
    exponential = math.exp(-Dt / tau)
    one_minus_exponential = 1.0 - exponential
 
    Q_plus = {}
    # 3.13  
    for t in range(T):
        solar_gain = I_t[t] * 0.15 # Affects penalty if *12 real
        internal_gain = 3 * 100
        heating = (x_sph_out_hp[t] + x_sph_out_boiler[t] + x_sph_out_fc[t] + x_sph_out_st[t])
        
        Q_plus[t] = solar_gain + internal_gain + heating
        
        
    Q_minus = 0
    
    # 3.13  
    model.addConstr(T_bdg[0] == (exponential * Tamb[0] + one_minus_exponential * Tamb[0] +
                                 (one_minus_exponential / U) * (Q_plus[0] - Q_minus)), name="building_temperature_rule_t0")

    # 3.13 
    for t in range(1, T):
        model.addConstr(T_bdg[t] == (exponential * T_bdg[t-1] + one_minus_exponential * Tamb[t] + 
                                     (one_minus_exponential / U) * (Q_plus[t] - Q_minus)), name=f"building_temperature_rule{t}")

    # 3.14  
    model.addConstrs((dT[t] >= T_bdg[t] - T_max for t in range(T)), name="comfort_upper_bound")

    # 3.14
    model.addConstrs((dT[t] >= T_min - T_bdg[t] for t in range(T)), name="comfort_lower_bound")
 
    
    print("BUILDING CONSTRAINTS OK")
     
    
    # NET FLOW HEAT TANK
    model.addConstrs((y_dhw_net_tank[t] == y_dhw_in_tank[t] - y_dhw_out_tank[t] for t in range(T)), name="dhw_net_flow_rule")

    # ELECTRICITY BALANCE
    model.addConstrs((x_el_out_fc[t] + x_el_out_pv[t] + y_el_out_battery[t] + el_grid_in[t] == 
        L_electricity[t] + y_el_in_battery[t] + x_el_in_hp[t] for t in range(T)), name="electricity_balance_rule")

    # SPACE HEATING BALANCE
    model.addConstrs((x_sph_out_fc[t] + x_sph_out_boiler[t] + x_sph_out_hp[t] + x_sph_out_st[t] == 
        L_sph[t] for t in range(T)), name="space_heating_balance_rule")

    # DHW BALANCE  
    model.addConstrs((x_dhw_out_fc[t] + x_dhw_out_boiler[t] + x_dhw_out_st[t] + y_dhw_out_tank[t] == 
        L_dhw[t] + y_dhw_in_tank[t] for t in range(T)), name="dhw_balance_rule")
    
    

    print("LOAD BALANCE CONSTRAINTS OK")

    # BINARY CONSTRAINTS 
    # Big_M 
    # Fuel Cell
    model.addConstr(x_gas_fc <= Big_M * binary_fc, "fc_ub")

    # PV
    model.addConstr(x_el_pv <= Big_M * binary_pv, name="pv_ub")

    # Solar Thermal
    model.addConstr(x_th_st <= Big_M * binary_st, name="st_ub")

    # Heat Pump
    model.addConstr(x_el_hp <= Big_M * binary_hp, name="hp_ub")

    # Boiler
    model.addConstr(x_gas_boiler <= Big_M * binary_boiler, name="boiler_ub")

    # Battery
    model.addConstr(y_el_battery <= Big_M_Joules * binary_battery, name="battery_ub")

    # Tank
    model.addConstr(y_h_tank <= Big_M_meters * binary_tank, name="tank_ub")
    
    # Grid
    #for t in range(T):
        #model.addConstr(el_grid_in[t] <= 10000, name=f"grid_capacity_limit_{t}" )

        
    # BINARY CONSTRAINTS 
    # Capacity
    # Fuel Cell 
    model.addConstr(x_gas_fc >= min_cap_fc * binary_fc, name="fc_lb")
    
    # PV
    model.addConstr(x_el_pv >= min_cap_pv * binary_pv, name="pv_lb")
    
    # Solar Thermal
    model.addConstr(x_th_st >= min_cap_st * binary_st, name="st_lb")
    
    # Heat Pump
    model.addConstr(x_el_hp >= min_cap_hp * binary_hp, name="hp_lb")
    
    # Boiler
    model.addConstr(x_gas_boiler >= min_cap_boiler * binary_boiler, name="boiler_lb")
    
    # Battery
    model.addConstr(y_el_battery >= min_cap_battery * binary_battery, name="battery_lb")
    
    # Tank
    model.addConstr(y_h_tank >= min_height_tank * binary_tank, name="tank_lb") 
    

    print("BINARY CONSTRAINTS OK")

        
######################################################################################################
    #CUTS
    model.addConstr(binary_boiler + binary_fc + binary_st >= 1, name="cut_1") # cuts boiler=0, fc=0, st=0
    
    model.addConstr(binary_boiler + binary_fc + binary_hp + binary_st >= 1, name="cut_2") # cuts boiler=0, fc=0, hp=0, st=0
    
    model.addConstr(binary_boiler + binary_fc + binary_hp + binary_st <= 3, name="cut_3") # cuts boiler=1, fc=1, hp=1, st=1
    
    model.addConstr(binary_tank <= binary_fc + binary_boiler + binary_st, name="cut_4") # cuts tank=1, fc=0, boiler=0, st=0
    
    model.addConstr(binary_boiler + binary_fc + binary_st <= 2 + (1 - binary_tank), name="cut_5") # cuts boiler=1, fc=1, st=1, tank=1
                
    model.addConstr(binary_fc + binary_pv <= 1 + binary_battery, name="cut_6") # cuts fc=1, pv=1, battery=0
    
######################################################################################################
    print("CUSTOM CUTS OK")
######################################################################################################
######################################################################################################
######################################################################################################
   

    # OBJECTIVE FUNCTION
    # 2.1a
    capex_fc = fc_Ccap * fc_h_el * (x_gas_fc / 1000.0)
    capex_pv = pv_Ccap * (x_el_pv / 1000.0)
    capex_st = st_Ccap * (x_th_st / 1000.0)
    capex_hp = hp_Ccap * hp_h_cop * (x_el_hp / 1000.0)    
    capex_boiler = boiler_Ccap * boiler_h * (x_gas_boiler / 1000.0)
    capex_battery = battery_Ccap * (y_el_battery / 3.6e6)
    capex_tank = heat_store_Ccap * (heat_store_capacity / 3.6e6)

    total_capex = (capex_fc + capex_pv + capex_st + capex_hp + capex_boiler + capex_battery + capex_tank)

    cost_gas_fc = (fc_Cfuel + co2_price * (fc_pco2 / 1000.0)) # 3.2
    final_cost_fc = sum(x_gas_in_fc[t] / 1000.0 * (Dt / 3600.0) * cost_gas_fc for t in range(T)) # 3.2
    
    cost_sph_hp = (hp_Cfuel + co2_price * (hp_pco2 / 1000.0)) # 3.7
    final_cost_hp = sum(x_el_in_hp[t] / 1000.0 * (Dt / 3600.0) * cost_sph_hp for t in range(T)) # 3.7
    
    cost_gas_boiler = (boiler_Cfuel + co2_price * (boiler_pco2 / 1000.0)) # 3.9
    final_cost_boiler = sum(x_gas_in_boiler[t] / 1000.0 * (Dt / 3600.0) * cost_gas_boiler for t in range(T)) # 3.9

    cost_grid_total = el_grid_cost + co2_price * (grid_pco2 / 1000.0)
    final_cost_grid = sum((el_grid_in[t] * Dt / 3.6e6) * cost_grid_total for t in range(T))

    total_final_cost = (final_cost_fc + final_cost_hp + final_cost_boiler + final_cost_grid)


    total_penalty = c_T * sum(dT[t] for t in range(T))

    # TOTAL OBJECTIVE
    model.setObjective(total_capex + total_final_cost + total_penalty + el_grid_connection_fee, GRB.MINIMIZE)

    print("OBJECTIVE OK")
 
    model.update()  

    # To get variables
    all_vars = model.getVars()
    num_vars = len(all_vars)
 
    lb = np.array(model.getAttr("LB", all_vars))
    ub = np.array(model.getAttr("UB", all_vars))

    vtypes = model.getAttr("VType", all_vars)

    # BINARY MADE CONTINUOUS BEFORE WITH LB AND UB CHOOSE THESE
    integer_var = [i >= num_vars-7 for i in range(num_vars)]

 
    return model, ub, lb, integer_var, num_vars, vtypes, binary_vars