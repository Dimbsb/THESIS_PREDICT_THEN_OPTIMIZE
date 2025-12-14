# EMS_model.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import math
from math import pi

def create_ems_model(T_hours=8760):
     
    model = gp.Model("EMS")
    model.Params.OutputFlag = 0   
    
    eps = np.finfo(float).eps
    
    # SOLAR RADIATION DATA FROM CSV FILE
    solar_data = pd.read_csv("SOLAR_DATA_OK.csv", parse_dates=["datetime"])
    solar_data = solar_data.set_index("datetime").resample("h").mean().iloc[:T_hours]
    I_t = solar_data["GHI"].values
    
    # ELECTRICITY DEMAND
    electricity_demand = pd.read_csv("RICHARDSON_OK.csv", parse_dates=["datetime"])
    electricity_demand = electricity_demand.set_index("datetime").resample("h").mean().iloc[:T_hours]
    L_electricity = electricity_demand["load_W"].values
    
    # DHW DEMAND 
    dhw_demand = pd.read_csv("DHW_OK.csv", parse_dates=["datetime"])
    dhw_demand = dhw_demand.set_index("datetime").resample("h").mean().iloc[:T_hours]
    L_dhw = dhw_demand["dhw_W"].values
    
    # SPACE HEAT DEMAND
    sph_demand = pd.read_csv("SPACE_HEAT_OK.csv", parse_dates=["datetime"])
    sph_demand = sph_demand.set_index("datetime").resample("h").mean().iloc[:T_hours]
    L_sph = sph_demand["Q_space_W"].values 
    
    # TEMPERATURE DATA
    temperature_demand = pd.read_csv("TEMPERATURES_OK.csv", parse_dates=["datetime"])
    temperature_demand = temperature_demand.set_index("datetime").resample("h").mean().iloc[:8760]
    Tamb = temperature_demand["Tamb_C"].values + 273.15
    Tcoll = temperature_demand["Tcoll_C"].values + 273.15
    
    print("ALL DATA LOADED")
    
    # Decision Variables
    # Fuel Cell
    x_gas_fc = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_gas_fc") 
    x_gas_in_fc = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_gas_in_fc")
    x_sph_out_fc = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_fc")    
    x_dhw_out_fc = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_fc")  
    x_el_out_fc = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_el_out_fc")
    
    # PV
    x_el_pv = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_el_pv")
    x_el_out_pv = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_el_out_pv")
    
    # Solar Thermal
    x_th_st = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_th_st")
    x_sph_out_st = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_st")     
    x_dhw_out_st = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_st") 

    # Heat Pump
    x_el_hp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_el_hp")
    x_sph_out_hp = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_st")
    x_el_in_hp = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_el_in_hp") 
    
    # Boiler
    x_gas_boiler = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_gas_boiler")
    x_gas_in_boiler = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_gas_in_boiler")  
    x_sph_out_boiler = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_sph_out_boiler")  
    x_dhw_out_boiler = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="x_dhw_out_boiler")
    
    # Battery
    y_el_battery = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_el_battery")
    y_el_in_battery = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="y_el_in_battery")   
    y_el_out_battery = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="y_el_out_battery")     
    
    # Heat Tank 
    y_h_tank = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_h_tank") 
    y_dhw_net_tank = model.addVars(T_hours, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_dhw_net_tank")
    y_dhw_in_tank = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="y_dhw_in_tank")   
    y_dhw_out_tank = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="y_dhw_out_tank") 
        
    # 3.13
    T_bdg = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="T_bdg") 

    # 3.14
    dT = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="dT") 

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
    battery_E = 0.95*(1.0/24.0) 
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
    heat_store_Thot = 60+273  
    heat_store_Tcold = 10+273  
    heat_store_Tbdg = 23+273  
    heat_store_yo = 0.50  
    heat_store_yT = 0.50  
    heat_store_mo = 0.50  

    # 3.13
    U = 168.0
    C = 15e6

    # 3.14 
    c_T = 10.0
    T_min = 18+273.15 
    T_max = 28+273.15

    # SUM
    SOC_el = model.addVars(T_hours, lb=0, vtype=GRB.CONTINUOUS, name="SOC_thermal")
    SOC_thermal = model.addVars(T_hours, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="SOC_thermal")

    Dt = 3600
    co2_price = 0.06

    print("PARAMETERS OK")

    # lower bounds 
    min_cap_fc = 1000.0       
    min_cap_pv = 1000.0       
    min_cap_st = 1000.0       
    min_cap_hp = 1000.0       
    min_cap_boiler = 1000.0   
    min_cap_battery = 1.0 * 3.6e6  
    min_height_tank = 0.5    

    # Big M  
    Big_M = 40000   
    Big_M_Joules = 1e10  
    Big_M_meters = 10.0  

    # Binary Decision Variables 1 or 0     
    binary_fc = model.addVar(vtype=GRB.BINARY, name="binary_fc")
    binary_pv = model.addVar(vtype=GRB.BINARY, name="binary_pv")
    binary_st = model.addVar(vtype=GRB.BINARY, name="binary_st")
    binary_hp = model.addVar(vtype=GRB.BINARY, name="binary_hp")
    binary_boiler = model.addVar(vtype=GRB.BINARY, name="binary_boiler")
    binary_battery = model.addVar(vtype=GRB.BINARY, name="binary_battery")
    binary_tank = model.addVar(vtype=GRB.BINARY, name="binary_tank")

    print("BINARY VARIABLES OK")
    
    for t in range(T_hours):
        # 3.1
        model.addConstr(fc_k * x_gas_fc <= x_gas_in_fc[t], name=f"fc_min_load{t}")
        # 3.1
        model.addConstr(x_gas_in_fc[t] <= x_gas_fc, name=f"fc_max_load{t}")
        # 3.1
        model.addConstr(x_el_out_fc[t] <= fc_h_el * x_gas_in_fc[t], name=f"fc_electricity_ub{t}")
        # 3.1
        model.addConstr(x_el_out_fc[t] >= eps, name=f"fc_electricity_lb{t}")
        # 3.1
        model.addConstr(x_sph_out_fc[t] + x_dhw_out_fc[t] <= fc_h_th * x_gas_in_fc[t], name=f"fc_heat_ub{t}")
    
        # 3.3  
        model.addConstr(x_el_out_pv[t] <= (pv_h_el * pv_h_pr * (I_t[t] / 1000.0) * (x_el_pv / pv_p)), name=f"pv_electricity_ub{t}")
        
        # 3.3  
        model.addConstr(x_el_out_pv[t] >= eps, name=f"pv_electricity_lb{t}")
        
        # 3.4  
        model.addConstr((x_sph_out_st[t] + x_dhw_out_st[t]) <= (((st_h * I_t[t]) - st_K * (Tcoll[t] - Tamb[t])) * x_th_st / st_p), name=f"st_heat_ub{t}")
        
        # 3.6 
        model.addConstr(x_sph_out_hp[t] <= hp_h_cop * x_el_hp, name=f"hp_space_heat_ub{t}")
        
        # 3.6 
        model.addConstr(x_sph_out_hp[t] >= eps, name=f"hp_space_heat_lb{t}")
        
        # 3.7 
        model.addConstr(x_el_in_hp[t] == x_sph_out_hp[t] / hp_h_cop, name=f"hp_electricity_consumption_rule{t}")
        
        # 3.8 
        model.addConstr(x_gas_in_boiler[t] <= x_gas_boiler, name=f"boiler_max_load{t}")
        
        # 3.8  
        model.addConstr((x_sph_out_boiler[t] + x_dhw_out_boiler[t]) <= boiler_h * x_gas_in_boiler[t], name=f"boiler_heat_ub{t}")
        
        # 3.8  
        model.addConstr(x_sph_out_boiler[t] + x_dhw_out_boiler[t] >= eps, name=f"boiler_heat_lb{t}")

    print("CONVERSION CONSTRAINTS OK")
      
      
    for t in range(T_hours):
        # 3.10
        model.addConstr(y_el_in_battery[t] <= battery_p_plus * y_el_battery, f"battery_charge_ub{t}")
        
        # 3.10
        model.addConstr(eps <= y_el_in_battery[t], f"battery_charge_lb{t}")
        
        # 3.10
        model.addConstr(y_el_out_battery[t] <= battery_p_minus * y_el_battery, f"battery_discharge_ub{t}")
        
        # 3.10
        model.addConstr(eps <= y_el_out_battery[t], f"battery_discharge_lb{t}")
 

    # 3.10
    model.addConstr(SOC_el[0] == battery_yo * y_el_battery, name="battery_soc_dynamics_zero")

    # 3.10
    for t in range(1, T_hours):
        model.addConstr(SOC_el[t] == (battery_E * SOC_el[t-1] + (battery_h_plus * y_el_in_battery[t] - 
                battery_h_minus * y_el_out_battery[t]) * Dt), name=f"battery_soc_dynamics{t}")

    # 3.10 
    model.addConstrs((battery_mo * y_el_battery <= SOC_el[t] for t in range(T_hours)), name="battery_soc_lower")

    # 3.10  
    model.addConstrs((SOC_el[t] <= y_el_battery for t in range(T_hours)), name="battery_soc_upper")

    # 3.10  
    model.addConstr(SOC_el[T_hours-1] == battery_yT * y_el_battery, "battery_soc_terminal")
    
    # 3.10 
    model.addConstr(sum(y_el_out_battery[t] for t in range(T_hours)) <= (T_hours / 24.0) * battery_h_minus * (1 - battery_mo) * y_el_battery, name="battery_cycling_limit")
 
    # 3.11  
    heat_store_capacity = (heat_store_p * heat_store_Cp * (heat_store_Thot - heat_store_Tcold) * pi * ((heat_store_F / 2.0)**2) * y_h_tank)

    # M1 3.11 
    heat_store_m1 = (pi * heat_store_U * (heat_store_Thot - heat_store_Tbdg) * (heat_store_F**2) / 2.0)

    # M2 3.11 
    heat_store_m2 = (pi * heat_store_U * (((heat_store_Thot - heat_store_Tcold)/2.0) - heat_store_Tbdg) * heat_store_F * y_h_tank)

    # 3.12  
    model.addConstr(SOC_thermal[0] == heat_store_yo * heat_store_capacity + y_dhw_net_tank[0] * Dt - (heat_store_m1 + heat_store_m2) * Dt, name="tank_soc_dynamics_t0")

    # 3.12  
    for t in range(1, T_hours):
        model.addConstr(SOC_thermal[t] == SOC_thermal[t-1] + y_dhw_net_tank[t] * Dt - (heat_store_m1 + heat_store_m2) * Dt, name=f"tank_soc_dynamics{t}")

    # 3.12  
    model.addConstrs((eps <= SOC_thermal[t] for t in range(T_hours)), name="tank_soc_lower")

    # 3.12  
    model.addConstrs((SOC_thermal[t] <= heat_store_capacity for t in range(T_hours)), name="tank_soc_upper")

    # 3.12  
    model.addConstr(SOC_thermal[T_hours-1] == heat_store_yT * heat_store_capacity, name="tank_soc_terminal")


    print("STORAGE CONSTRAINTS OK")
    
 
    # 3.13  
    tau = C / U
    exponential = math.exp(-Dt / tau)
    one_minus_exponential = 1.0 - exponential
 

    Q_plus = {}
    # 3.13  
    for t in range(T_hours):
        solar_gain = I_t[t] * 0.15
        internal_gain = 3 * 100
        tank_losses = heat_store_m1 + heat_store_m2
        heating = (x_sph_out_hp[t] + x_sph_out_boiler[t] + x_sph_out_fc[t] + x_sph_out_st[t])
        
        Q_plus[t] = solar_gain + internal_gain - tank_losses + heating
        
        
    Q_minus = 0
    
    # 3.13   
    model.addConstr(T_bdg[0] == Tamb[0], "building_temperature_rule_t0")
 
    # 3.13 
    for t in range(1, T_hours):
        model.addConstr(T_bdg[t] == (exponential * T_bdg[t-1] + one_minus_exponential * Tamb[t] + 
                                     (one_minus_exponential / U) * (Q_plus[t] - Q_minus)), name=f"building_temperature_rule{t}")

    # 3.14  
    model.addConstrs((dT[t] >= T_bdg[t] - T_max for t in range(T_hours)), name="comfort_lower_bound")

    # 3.14
    model.addConstrs((dT[t] >= T_min - T_bdg[t] for t in range(T_hours)), name="comfort_upper_bound")

    # 3.14
    model.addConstrs((dT[t] >= 0 for t in range(T_hours)), name="comfort_non_negative_rule")
    
    
    print("BUILDING CONSTRAINTS OK")
     
    
    # NET FLOW HEAT TANK
    model.addConstrs((y_dhw_net_tank[t] == y_dhw_in_tank[t] - y_dhw_out_tank[t] for t in range(T_hours)), name="dhw_net_flow_rule")

    # ELECTRICITY BALANCE
    model.addConstrs((x_el_out_fc[t] + x_el_out_pv[t] + y_el_out_battery[t] == 
        L_electricity[t] + y_el_in_battery[t] for t in range(T_hours)), name="electricity_balance_rule")

    # SPACE HEATING BALANCE
    model.addConstrs((x_sph_out_fc[t] + x_sph_out_boiler[t] + x_sph_out_hp[t] + x_sph_out_st[t] == 
        L_sph[t] for t in range(T_hours)), name="space_heating_balance_rule")

    # DHW BALANCE - ΔΙΟΡΘΩΜΕΝΟ (Pyomo είχε λάθος!)
    model.addConstrs((x_dhw_out_fc[t] + x_dhw_out_boiler[t] + x_dhw_out_st[t] + y_dhw_out_tank[t] == 
        L_dhw[t] + y_dhw_in_tank[t] for t in range(T_hours)), name="dhw_balance_rule")
    
    

    print("LOAD BALANCE CONSTRAINTS OK")

    
     
    # BINARY CONSTRAINTS 
    # Big_M 
    # Fuel Cell
    model.addConstr(x_gas_fc <= Big_M * binary_fc, "fc_bin_ub")

    # PV
    model.addConstr(x_el_pv <= Big_M * binary_pv, name="pv_bin_ub")

    # Solar Thermal
    model.addConstr(x_th_st <= Big_M * binary_st, name="st_bin_ub")

    # Heat Pump
    model.addConstr(x_el_hp <= Big_M * binary_hp, name="hp_bin_ub")

    # Boiler
    model.addConstr(x_gas_boiler <= Big_M * binary_boiler, name="boiler_bin_ub")

    # Battery
    model.addConstr(y_el_battery <= Big_M_Joules * binary_battery, name="battery_bin_ub")

    # Tank
    model.addConstr(y_h_tank <= Big_M_meters * binary_tank, name="tank_bin_ub")

        
    # BINARY CONSTRAINTS 
    # Capacity
    # Fuel Cell 
    
    model.addConstr(x_gas_fc >= min_cap_fc * binary_fc, name="fc_bin_lb")
    model.addConstr(x_el_pv >= min_cap_pv * binary_pv, name="pv_bin_lb")
    model.addConstr(x_th_st >= min_cap_st * binary_st, name="st_bin_lb")
    model.addConstr(x_el_hp >= min_cap_hp * binary_hp, name="hp_bin_lb")
    model.addConstr(x_gas_boiler >= min_cap_boiler * binary_boiler, name="boiler_bin_lb")
    model.addConstr(y_el_battery >= min_cap_battery * binary_battery, name="battery_bin_lb")
    model.addConstr(y_h_tank >= min_height_tank * binary_tank, name="tank_bin_lb") 
     
    
    print("BINARY CONSTRAINTS OK")
    
    
    
    # OBJECTIVE FUNCTION
    # 2.1a
    capex_fc = fc_Ccap * fc_h_el * (x_gas_fc / 1000.0)
    capex_pv = pv_Ccap * (x_el_pv / 1000.0)
    capex_st = st_Ccap * (x_th_st / 1000.0)
    capex_hp = hp_Ccap * (x_el_hp / 1000.0)
    capex_boiler = boiler_Ccap * boiler_h * (x_gas_boiler / 1000.0)
    capex_battery = battery_Ccap * (y_el_battery / 3.6e6)
    capex_tank = heat_store_Ccap * (heat_store_capacity / 3.6e6)

    total_capex = (capex_fc + capex_pv + capex_st + capex_hp + capex_boiler + capex_battery + capex_tank)

    cost_gas_fc = fc_Cfuel + co2_price * (fc_pco2 / 1000.0) # 3.2
    final_cost_fc = gp.quicksum(x_gas_in_fc[t] / 1000.0 * cost_gas_fc for t in range(T_hours)) # 3.2

    cost_sph_hp = hp_Cfuel + co2_price * (hp_pco2 / 1000.0) # 3.7
    final_cost_hp = gp.quicksum(x_el_in_hp[t] / 1000.0 * cost_sph_hp for t in range(T_hours)) # 3.7

    cost_gas_boiler = boiler_Cfuel + co2_price * (boiler_pco2 / 1000.0) # 3.9
    final_cost_boiler = gp.quicksum(x_gas_in_boiler[t] / 1000.0 * cost_gas_boiler for t in range(T_hours)) # 3.9

    total_final_cost = final_cost_fc + final_cost_hp + final_cost_boiler

    total_penalty = c_T * gp.quicksum(dT[t] for t in range(T_hours))

    # TOTAL OBJECTIVE
    model.setObjective(total_capex + total_final_cost + total_penalty, GRB.MINIMIZE)

    print("OBJECTIVE OK")
 

    
    model.update()  

    # To get variables
    all_vars = model.getVars()
    num_vars = len(all_vars)
 
    lb = np.array(model.getAttr("LB", all_vars))
    ub = np.array(model.getAttr("UB", all_vars))

    vtypes = model.getAttr("VType", all_vars)

    integer_var = np.array([vt in ['B', 'I'] for vt in vtypes], dtype=bool)
 
    return model, ub, lb, integer_var, num_vars