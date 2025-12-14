#A linear programming approach to the optimization of residential energy systems
import pyomo.environ as pyomo 
import numpy as np
import pandas as pd
import math
from math import pi
import time
from pyomo.environ import SolverFactory

# initialize the model
model = pyomo.ConcreteModel()
# Time 
model.T = pyomo.Set(initialize=range(8760)) 
eps = np.finfo(float).eps

# SOLAR RADIATION DATA FROM CSV FILE
solar_data = pd.read_csv("SOLAR_DATA_OK.csv", parse_dates=["datetime"]) 
solar_data = solar_data.set_index("datetime").resample("h").mean()
ghi_hourly_values = solar_data["GHI"].values[:8760]

def radiation_initialize(model, t):
    return float(ghi_hourly_values[t])
model.I_t = pyomo.Param(model.T, initialize=radiation_initialize, domain=pyomo.NonNegativeReals)


# ELECTRICITY DEMAND  
electricity_demand = pd.read_csv("RICHARDSON_OK.csv", parse_dates=["datetime"])
electricity_demand = electricity_demand.set_index("datetime").resample("h").mean()
electricity_load = electricity_demand["load_W"].values[:8760]

def L_electricity_initialize(model, t):
    return float(electricity_load[t])
model.L_electricity = pyomo.Param(model.T, initialize=L_electricity_initialize,domain=pyomo.NonNegativeReals)


# DHW DEMAND 
dhw_demand = pd.read_csv("DHW_OK.csv", parse_dates=["datetime"])
dhw_demand = dhw_demand.set_index("datetime").resample("h").mean()
dhw_load = dhw_demand["dhw_W"].values[:8760]

def L_dhw_initialize(model, t):
    return float(dhw_load[t])
model.L_dhw = pyomo.Param(model.T, initialize=L_dhw_initialize, domain=pyomo.NonNegativeReals)


# SPACE HEAT DEMAND
sph_demand = pd.read_csv("SPACE_HEAT_OK.csv", parse_dates=["datetime"])
sph_demand = sph_demand.set_index("datetime").resample("h").mean()
sph_load = sph_demand["Q_space_W"].values[:8760]

def L_sph_initialize(model, t):
    return float(sph_load[t])
model.L_sph = pyomo.Param(model.T, initialize=L_sph_initialize, domain=pyomo.NonNegativeReals)


# TEMPERATURE DATA
temperature_demand = pd.read_csv("TEMPERATURES_OK.csv", parse_dates=["datetime"])
temperature_demand = temperature_demand.set_index("datetime").resample("h").mean() 
Tamb_series = temperature_demand["Tamb_C"].values[:8760]
Tcoll_series = temperature_demand["Tcoll_C"].values[:8760]

def Tamb_initialize(m, t):
    return float(Tamb_series[t]) + 273.15   
model.T_amb = pyomo.Param(model.T, initialize=Tamb_initialize, domain=pyomo.Reals)

def Tcoll_initialize(m, t):
    return float(Tcoll_series[t]) + 273.15 
model.T_coll = pyomo.Param(model.T, initialize=Tcoll_initialize, domain=pyomo.Reals)

print("ALL DATA LOADED")

# Decision Variables
# Fuel Cell
model.x_gas_fc = pyomo.Var(domain=pyomo.NonNegativeReals) 
model.x_gas_in_fc = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)      
model.x_sph_out_fc = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)     
model.x_dhw_out_fc = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)     
model.x_el_out_fc = pyomo.Var(model.T, domain=pyomo.NonNegativeReals) 

# PV
model.x_el_pv = pyomo.Var(domain=pyomo.NonNegativeReals) 
model.x_el_out_pv = pyomo.Var(model.T, domain=pyomo.NonNegativeReals) 

# Solar Thermal
model.x_th_st = pyomo.Var(domain=pyomo.NonNegativeReals) 
model.x_sph_out_st = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)      
model.x_dhw_out_st = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  

# Heat Pump
model.x_el_hp = pyomo.Var(domain=pyomo.NonNegativeReals)  
model.x_sph_out_hp = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.x_el_in_hp = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Boiler
model.x_gas_boiler = pyomo.Var(domain=pyomo.NonNegativeReals) 
model.x_gas_in_boiler = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)   
model.x_sph_out_boiler = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  
model.x_dhw_out_boiler = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  

# Battery
model.y_el_battery = pyomo.Var(domain=pyomo.NonNegativeReals) 
model.y_el_in_battery = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)    
model.y_el_out_battery = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)   

# Heat Tank  
model.y_h_tank = pyomo.Var(domain=pyomo.NonNegativeReals)  
model.y_dhw_net_tank = pyomo.Var(model.T, domain=pyomo.Reals) 
model.y_dhw_in_tank = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)   
model.y_dhw_out_tank = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# 3.13
model.T_bdg = pyomo.Var(model.T, domain=pyomo.PositiveReals)

# 3.14
model.dT = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

print("DECISION VARIABLES OK")

# Parameters - Table 4 - Table 5
# Conversion Technologies

# Fuel Cell Parameters
model.fc_Cinv = pyomo.Param(initialize=4000)   
model.fc_Comex = pyomo.Param(initialize=200)   
model.fc_Cfuel = pyomo.Param(initialize=0.1318)  
model.fc_Tlife = pyomo.Param(initialize=15)   
model.fc_Ccap = pyomo.Param(initialize=535)   
model.fc_h_el = pyomo.Param(initialize=0.50)   
model.fc_h_th = pyomo.Param(initialize=0.40)   
model.fc_pco2 = pyomo.Param(initialize=200)   
model.fc_k = pyomo.Param(initialize=0.30)   

# PV Parameters
model.pv_Cinv = pyomo.Param(initialize=2500)  
model.pv_Comex = pyomo.Param(initialize=36)   
model.pv_Tlife = pyomo.Param(initialize=25)   
model.pv_Ccap = pyomo.Param(initialize=180)   
model.pv_h_el = pyomo.Param(initialize=0.16)   
model.pv_h_pr = pyomo.Param(initialize=0.90)   
model.pv_p = pyomo.Param(initialize=144)   

# Solar Thermal Parameters
model.st_Cinv = pyomo.Param(initialize=770)  
model.st_Comex = pyomo.Param(initialize=28)   
model.st_Tlife = pyomo.Param(initialize=20)   
model.st_Ccap = pyomo.Param(initialize=80)   
model.st_p = pyomo.Param(initialize=650)   
model.st_h = pyomo.Param(initialize=0.80)   
model.st_K = pyomo.Param(initialize=4.0)   

# Heat Pump Parameters
model.hp_Cinv = pyomo.Param(initialize=1500)  
model.hp_Comex = pyomo.Param(initialize=99)   
model.hp_Tlife = pyomo.Param(initialize=23)   
model.hp_Ccap = pyomo.Param(initialize=190)   
model.hp_h_cop = pyomo.Param(initialize=4)   
model.hp_Cfuel = pyomo.Param(initialize=0.0555)   
model.hp_pco2 = pyomo.Param(initialize=91.5)   

# Boiler Parameters
model.boiler_Cinv = pyomo.Param(initialize=178)  
model.boiler_Comex = pyomo.Param(initialize=14)   
model.boiler_Cfuel = pyomo.Param(initialize=0.1186)   
model.boiler_Tlife = pyomo.Param(initialize=17)   
model.boiler_Ccap = pyomo.Param(initialize=28)   
model.boiler_h = pyomo.Param(initialize=0.95)  
model.boiler_pco2 = pyomo.Param(initialize=179)   

# Storage Technologies
# Battery Parameters
model.battery_Cinv = pyomo.Param(initialize=1000)   
model.battery_Tlife = pyomo.Param(initialize=9)   
model.battery_Ccap = pyomo.Param(initialize=128)   
model.battery_h = pyomo.Param(initialize=0.92)   
model.battery_E = pyomo.Param(initialize=0.95*(1.0/24.0))  
model.battery_mo = pyomo.Param(initialize=0.20)   
model.battery_p_plus = pyomo.Param(initialize=0.58)   
model.battery_p_minus = pyomo.Param(initialize=1.00)   
model.battery_yo  = pyomo.Param(initialize=1.00)  
model.battery_yT = pyomo.Param(initialize=1.00)  
# 3.10
model.battery_h_plus = pyomo.Param(initialize=np.sqrt(model.battery_h.value))
model.battery_h_minus = pyomo.Param(initialize=1.0/np.sqrt(model.battery_h.value))

# Heat Store Parameters
model.heat_store_Cinv = pyomo.Param(initialize=86)  
model.heat_store_Tlife = pyomo.Param(initialize=20)  
model.heat_store_Ccap = pyomo.Param(initialize=6)  
model.heat_store_F = pyomo.Param(initialize=0.80)  
model.heat_store_U = pyomo.Param(initialize=0.21)  
model.heat_store_Cp = pyomo.Param(initialize=4.19e3)    
model.heat_store_p = pyomo.Param(initialize=1000)   
model.heat_store_Thot = pyomo.Param(initialize=60+273)   
model.heat_store_Tcold = pyomo.Param(initialize=10+273)   
model.heat_store_Tbdg = pyomo.Param(initialize=23+273)   
model.heat_store_yo = pyomo.Param(initialize=0.50)   
model.heat_store_yT = pyomo.Param(initialize=0.50)   
model.heat_store_mo = pyomo.Param(initialize=0.50)   

# 3.13
model.U = pyomo.Param(initialize=168.0)
model.C = pyomo.Param(initialize=15e6)

# 3.14 
model.c_T = pyomo.Param(initialize=10.0)
model.T_min = pyomo.Param(initialize=18+273.15)  
model.T_max = pyomo.Param(initialize=28+273.15) 

# SUM
model.SOC_el = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.SOC_thermal = pyomo.Var(model.T, domain=pyomo.Reals)

model.Dt = pyomo.Param(initialize=3600)   
model.co2_price = pyomo.Param(initialize=0.06)

print("PARAMETERS OK")

# lower bounds 
min_cap_fc = 1000.0       
min_cap_pv = 1000.0       
min_cap_st = 1000.0       
min_cap_hp = 1000.0       
min_cap_boiler = 1000.0   
min_cap_batt = 1.0 * 3.6e6  
min_height_tank = 0.5    

# Big M  
Big_M = 40000   
Big_M_Joules = 1e10  
Big_M_meters = 10.0  

# Binary Decision Variables 1 or 0 
model.binary_fc = pyomo.Var(domain=pyomo.Binary)
model.binary_pv = pyomo.Var(domain=pyomo.Binary)
model.binary_st = pyomo.Var(domain=pyomo.Binary)
model.binary_hp = pyomo.Var(domain=pyomo.Binary)
model.binary_boiler = pyomo.Var(domain=pyomo.Binary)
model.binary_batt = pyomo.Var(domain=pyomo.Binary)
model.binary_tank = pyomo.Var(domain=pyomo.Binary)

print("BINARY VARIABLES OK")

# 3.1 
def fc_min_load(model, t):
    return model.fc_k * model.x_gas_fc <= model.x_gas_in_fc[t]
model.constraint_fc_min = pyomo.Constraint(model.T, rule=fc_min_load)

# 3.1  
def fc_max_load(model, t):
    return model.x_gas_in_fc[t] <= model.x_gas_fc
model.constraint_fc_max = pyomo.Constraint(model.T, rule=fc_max_load)

# 3.1  
def fc_electricity_ub(model, t):
    return model.x_el_out_fc[t] <= model.fc_h_el * model.x_gas_in_fc[t]
model.constraint_fc_elec = pyomo.Constraint(model.T, rule=fc_electricity_ub)

#3.1  
def fc_electricity_lb(model, t):
    return  eps<= model.x_el_out_fc[t]  
model.constraint_fc_elec_lb = pyomo.Constraint(model.T, rule=fc_electricity_lb)

# 3.1 
def fc_heat_ub(model, t):
        return (model.x_sph_out_fc[t] + model.x_dhw_out_fc[t]) <= model.fc_h_th * model.x_gas_in_fc[t]
model.constraint_fc_heat = pyomo.Constraint(model.T, rule=fc_heat_ub)

# 3.3  
def pv_electricity_ub(model, t):
    return model.x_el_out_pv[t] <= (model.pv_h_el * model.pv_h_pr * (model.I_t[t] / 1000.0) * (model.x_el_pv / model.pv_p))
model.constraint_pv_elec = pyomo.Constraint(model.T, rule=pv_electricity_ub)

# 3.3  
def pv_electricity_lb(model, t):
    return eps <= model.x_el_out_pv[t]  
model.constraint_pv_elec_lb = pyomo.Constraint(model.T, rule=pv_electricity_lb)

# 3.4 
def st_heat_ub(model, t):
    return (model.x_sph_out_st[t] + model.x_dhw_out_st[t]) <= (((model.st_h * model.I_t[t]) - model.st_K * (model.T_coll[t] - model.T_amb[t])) * (model.x_th_st)) / model.st_p
model.constraint_st_heat = pyomo.Constraint(model.T, rule=st_heat_ub)

# 3.6 
def hp_space_heat_ub(model, t):
    return model.x_sph_out_hp[t] <= model.hp_h_cop * model.x_el_hp
model.constraint_hp_sph = pyomo.Constraint(model.T, rule=hp_space_heat_ub)

# 3.6 
def hp_space_heat_lb(model, t):
    return  eps <= model.x_sph_out_hp[t]   
model.constraint_hp_sph_lb = pyomo.Constraint(model.T, rule=hp_space_heat_lb)

# 3.7  
def hp_electricity_consumption_rule(model, t):
    return model.x_el_in_hp[t] == model.x_sph_out_hp[t] / model.hp_h_cop
model.constraint_hp_elec_consumption = pyomo.Constraint(model.T, rule=hp_electricity_consumption_rule)

# 3.8 
def boiler_max_load(model, t):
    return model.x_gas_in_boiler[t] <= model.x_gas_boiler
model.constraint_boiler_max = pyomo.Constraint(model.T, rule=boiler_max_load)

# 3.8 
def boiler_heat_ub(model, t):
    return (model.x_sph_out_boiler[t] + model.x_dhw_out_boiler[t]) <= model.boiler_h * model.x_gas_in_boiler[t]
model.constraint_boiler_heat = pyomo.Constraint(model.T, rule=boiler_heat_ub)

# 3.8 
def boiler_heat_lb(model, t):
    return eps <= (model.x_sph_out_boiler[t] + model.x_dhw_out_boiler[t])  
model.constraint_boiler_heat_lb = pyomo.Constraint(model.T, rule=boiler_heat_lb)

print("CONVERSION CONSTRAINTS OK")

# 3.10 
def battery_charge_ub(model, t):
    return model.y_el_in_battery[t] <= model.battery_p_plus * model.y_el_battery
model.constraint_battery_ch_ub = pyomo.Constraint(model.T, rule=battery_charge_ub)

# 3.10 
def battery_charge_lb(model, t):
    return eps <= model.y_el_in_battery[t]
model.constraint_battery_ch_lb = pyomo.Constraint(model.T, rule=battery_charge_lb)

# 3.10 
def battery_discharge_ub(model, t):
    return model.y_el_out_battery[t] <= model.battery_p_minus * model.y_el_battery
model.constraint_battery_dis_ub = pyomo.Constraint(model.T, rule=battery_discharge_ub)

# 3.10 
def battery_discharge_lb(model, t):
    return eps <= model.y_el_out_battery[t]
model.constraint_battery_dis_lb = pyomo.Constraint(model.T, rule=battery_discharge_lb)

# 3.10 
def battery_soc_dynamics(model, t):
    if t == 0:
        return model.SOC_el[0] == model.battery_yo * model.y_el_battery
    else:
        return (model.SOC_el[t] == 
                model.battery_E * model.SOC_el[t-1] + 
                (model.battery_h_plus * model.y_el_in_battery[t] - 
                 model.battery_h_minus * model.y_el_out_battery[t]) * 
                model.Dt)

model.battery_soc_constraint = pyomo.Constraint(model.T, rule=battery_soc_dynamics)

# 3.10 
def battery_soc_lower(model, t):
    return model.battery_mo * model.y_el_battery <= model.SOC_el[t]
model.battery_soc_lb = pyomo.Constraint(model.T, rule=battery_soc_lower)

# 3.10 
def battery_soc_upper(model, t):
    return model.SOC_el[t] <= model.y_el_battery
model.battery_soc_ub = pyomo.Constraint(model.T, rule=battery_soc_upper)

# 3.10 
def battery_soc_terminal(model):
    t=model.T.last()
    return model.SOC_el[t] == model.battery_yT * model.y_el_battery
model.constraint_battery_soc_term = pyomo.Constraint(rule=battery_soc_terminal)

# 3.10 
def battery_cycling_limit(model):
    T = len(model.T)
    return (
        sum(model.y_el_out_battery[t] for t in model.T)
        <= (T / 24.0) * model.battery_h_minus * (1 - model.battery_mo) * model.y_el_battery)
model.constraint_battery_cycling = pyomo.Constraint(rule=battery_cycling_limit)

# 3.11 
def heat_store_capacity_rule(model):
    return (model.heat_store_p * (model.heat_store_Cp) * 
        (model.heat_store_Thot - model.heat_store_Tcold) * pi * ((model.heat_store_F / 2.0)**2) * model.y_h_tank)    
model.heat_store_capacity = pyomo.Expression(rule=heat_store_capacity_rule)

# M1 3.11 
def heat_store_m1_rule(model):
    return (pi * model.heat_store_U * (model.heat_store_Thot - model.heat_store_Tbdg) * 
            (model.heat_store_F**2) / 2.0)
model.heat_store_m1 = pyomo.Expression(rule=heat_store_m1_rule)

# M2 3.11 
def heat_store_m2_rule(model):
    return (pi * model.heat_store_U * (((model.heat_store_Thot - model.heat_store_Tcold)/2.0) - 
            model.heat_store_Tbdg) * model.heat_store_F * model.y_h_tank)
model.heat_store_m2 = pyomo.Expression(rule=heat_store_m2_rule)

# 3.12 
def tank_soc_dynamics(model, t):
    if t == 0:
        return (model.SOC_thermal[0] == 
                model.heat_store_yo * model.heat_store_capacity + 
                model.y_dhw_net_tank[0] * model.Dt - 
                (model.heat_store_m1 + model.heat_store_m2) * model.Dt)
    else:
        return (model.SOC_thermal[t] == 
                model.SOC_thermal[t-1] + 
                model.y_dhw_net_tank[t] * model.Dt - 
                (model.heat_store_m1 + model.heat_store_m2) * model.Dt)

model.tank_soc_constraint = pyomo.Constraint(model.T, rule=tank_soc_dynamics)

# 3.12 
def tank_soc_lower(model, t):
    return eps <= model.SOC_thermal[t]
model.constraint_tank_soc_lb = pyomo.Constraint(model.T, rule=tank_soc_lower)

# 3.12 
def tank_soc_upper(model, t):
    return model.SOC_thermal[t] <= model.heat_store_capacity
model.constraint_tank_soc_ub = pyomo.Constraint(model.T, rule=tank_soc_upper)

# 3.12 
def tank_soc_terminal(model):
    t= model.T.last()
    return model.SOC_thermal[t] == model.heat_store_yT * model.heat_store_capacity
model.constraint_tank_soc_term = pyomo.Constraint(rule=tank_soc_terminal)

print("STORAGE CONSTRAINTS OK")

# 3.13 
def tau_initialize(model):
    return model.C / model.U
model.tau = pyomo.Param(initialize=tau_initialize)

# 3.13 
def exponential_initialize(model):
    return math.exp((-model.Dt) / model.tau)
model.exponential = pyomo.Param(initialize=exponential_initialize)

# 3.13 
def one_minus_exponential_initialize(model):
    return 1.0 - model.exponential
model.one_minus_exponential = pyomo.Param(initialize=one_minus_exponential_initialize)


# 3.13 
# Q_plus
def Q_plus(model, t):
    
    solar_gain = model.I_t[t] * 0.15
    internal_gain = 3 * 100
    
    # Heat Store Losses
    tank_losses = model.heat_store_m1 + model.heat_store_m2
    
    heating = (model.x_sph_out_hp[t] + model.x_sph_out_boiler[t] + model.x_sph_out_fc[t] + model.x_sph_out_st[t])
    
    return solar_gain + internal_gain - tank_losses + heating
model.Q_plus = pyomo.Expression(model.T, rule=Q_plus)


# 3.13 
# Q_minus
def Q_minus(model, t):
    return 0.0
model.Q_minus = pyomo.Expression(model.T, rule=Q_minus)


# 3.13 
def building_temperature_rule(model, t):
    if t == 0:
        return model.T_bdg[t] == model.T_amb[t]
    else:
        return (model.T_bdg[t] ==
                model.exponential * model.T_bdg[t - 1] + 
                model.one_minus_exponential * model.T_amb[t] +
                (model.one_minus_exponential / model.U) * (model.Q_plus[t] - model.Q_minus[t]))
model.building_temperature = pyomo.Constraint(model.T, rule=building_temperature_rule)

# 3.14 
def comfort_lower_bound(model, t):
    return model.dT[t] >= model.T_bdg[t] - model.T_max
model.constraint_comfort_lower = pyomo.Constraint(model.T, rule=comfort_lower_bound)

# 3.14 
def comfort_upper_bound(model, t):
    return model.dT[t] >= model.T_min - model.T_bdg[t]
model.constraint_comfort_upper = pyomo.Constraint(model.T, rule=comfort_upper_bound)

# 3.14 
def comfort_non_negative_rule(model, t):
    return model.dT[t] >= 0
model.constraint_comfort_non_negative = pyomo.Constraint(model.T, rule=comfort_non_negative_rule)

print("BUILDING CONSTRAINTS OK")


# NET FLOW HEAT TANK
def dhw_net_flow_rule(model, t):
    return model.y_dhw_net_tank[t] == model.y_dhw_in_tank[t] - model.y_dhw_out_tank[t]
model.constraint_dhw_net = pyomo.Constraint(model.T, rule=dhw_net_flow_rule)

# ELECTRICITY BALANCE
def electricity_balance_rule(model, t):
    supply = (model.x_el_out_fc[t] + model.x_el_out_pv[t] + model.y_el_out_battery[t])
    demand = model.L_electricity[t] + model.y_el_in_battery[t]
    return supply == demand
model.constraint_electricity_balance = pyomo.Constraint(model.T, rule=electricity_balance_rule)

# SPACE HEATING BALANCE
def space_heating_balance_rule(model, t):
    supply = (model.x_sph_out_fc[t] + model.x_sph_out_boiler[t] + 
              model.x_sph_out_hp[t] + model.x_sph_out_st[t])
    demand = model.L_sph[t]
    return supply == demand
model.constraint_space_heating_balance = pyomo.Constraint(model.T, rule=space_heating_balance_rule)

# DHW BALANCE
def dhw_balance_rule(model, t):
    supply = (model.x_dhw_out_fc[t] + model.x_dhw_out_boiler[t] + model.x_dhw_out_st[t] + model.y_dhw_out_tank[t])
    demand = model.L_dhw[t] + model.y_dhw_in_tank[t] 
    return supply - model.y_dhw_in_tank[t] == demand
model.constraint_dhw_balance = pyomo.Constraint(model.T, rule=dhw_balance_rule)


print("LOAD BALANCE CONSTRAINTS OK")


# BINARY CONSTRAINTS 
# Big_M 
# Fuel Cell
def fc_bin_ub(model): 
    return model.x_gas_fc <= Big_M * model.binary_fc
model.c_fc_bin_ub = pyomo.Constraint(rule=fc_bin_ub)

# PV
def pv_bin_ub(model): 
    return model.x_el_pv <= Big_M * model.binary_pv
model.c_pv_bin_ub = pyomo.Constraint(rule=pv_bin_ub)

# Solar Thermal
def st_bin_ub(model): 
    return model.x_th_st <= Big_M * model.binary_st
model.c_st_bin_ub = pyomo.Constraint(rule=st_bin_ub)

# Heat Pump
def hp_bin_ub(model): 
    return model.x_el_hp <= Big_M * model.binary_hp
model.c_hp_bin_ub = pyomo.Constraint(rule=hp_bin_ub)

# Boiler
def boiler_bin_ub(model): 
    return model.x_gas_boiler <= Big_M * model.binary_boiler
model.c_boiler_bin_ub = pyomo.Constraint(rule=boiler_bin_ub)

# Battery
def batt_bin_ub(model): 
    return model.y_el_battery <= Big_M_Joules * model.binary_batt
model.c_batt_bin_ub = pyomo.Constraint(rule=batt_bin_ub)

# Tank
def tank_bin_ub(model): 
    return model.y_h_tank <= Big_M_meters * model.binary_tank
model.c_tank_bin_ub = pyomo.Constraint(rule=tank_bin_ub)



# BINARY CONSTRAINTS 
# Capacity
# Fuel Cell
def fc_bin_lb(model): 
    return model.x_gas_fc >= min_cap_fc * model.binary_fc
model.c_fc_bin_lb = pyomo.Constraint(rule=fc_bin_lb)

# PV
def pv_bin_lb(model): 
    return model.x_el_pv >= min_cap_pv * model.binary_pv
model.c_pv_bin_lb = pyomo.Constraint(rule=pv_bin_lb)

# Solar Thermal
def st_bin_lb(model): 
    return model.x_th_st >= min_cap_st * model.binary_st
model.c_st_bin_lb = pyomo.Constraint(rule=st_bin_lb)

# Heat Pump
def hp_bin_lb(model): 
    return model.x_el_hp >= min_cap_hp * model.binary_hp
model.c_hp_bin_lb = pyomo.Constraint(rule=hp_bin_lb)

# Boiler
def boiler_bin_lb(model): 
    return model.x_gas_boiler >= min_cap_boiler * model.binary_boiler
model.c_boiler_bin_lb = pyomo.Constraint(rule=boiler_bin_lb)

# Battery
def batt_bin_lb(model): 
    return model.y_el_battery >= min_cap_batt * model.binary_batt
model.c_batt_bin_lb = pyomo.Constraint(rule=batt_bin_lb)

# Tank
def tank_bin_lb(model): 
    return model.y_h_tank >= min_height_tank * model.binary_tank
model.c_tank_bin_lb = pyomo.Constraint(rule=tank_bin_lb)


print("BINARY CONSTRAINTS OK")




# OBJECTIVE FUNCTION
# 2.1a
def objective_rule(model):
 
    capex_fc      = model.fc_Ccap * model.fc_h_el * (model.x_gas_fc / 1000.0)
    capex_pv      = model.pv_Ccap * (model.x_el_pv / 1000.0)
    capex_st      = model.st_Ccap * (model.x_th_st / 1000.0)
    capex_hp      = model.hp_Ccap * (model.x_el_hp / 1000.0)
    capex_boiler  = model.boiler_Ccap * model.boiler_h *(model.x_gas_boiler / 1000.0)
    capex_battery = model.battery_Ccap * (model.y_el_battery / 3.6e6)  
    capex_tank    = model.heat_store_Ccap * (model.heat_store_capacity / 3.6e6)
    
    total_capex = (capex_fc + capex_pv + capex_st + capex_hp + capex_boiler + capex_battery + capex_tank)

 
    cost_gas_fc = (model.fc_Cfuel + model.co2_price * (model.fc_pco2 / 1000.0)) # 3.2
    final_cost_fc = sum(model.x_gas_in_fc[t] / 1000.0 * cost_gas_fc for t in model.T) # 3.2
    
    cost_sph_hp = (model.hp_Cfuel + model.co2_price * (model.hp_pco2 / 1000.0)) # 3.7
    final_cost_hp = sum(model.x_el_in_hp[t] / 1000.0 * cost_sph_hp for t in model.T) # 3.7
    
    cost_gas_boiler = (model.boiler_Cfuel + model.co2_price * (model.boiler_pco2 / 1000.0)) # 3.9
    final_cost_boiler = sum(model.x_gas_in_boiler[t] / 1000.0 * cost_gas_boiler for t in model.T) # 3.9
    
 
    total_final_cost = final_cost_fc + final_cost_hp + final_cost_boiler 
 
    penalty = model.c_T * sum(model.dT[t] for t in model.T)

    return total_capex + total_final_cost + penalty

model.objective = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

print("OBJECTIVE OK")
 

# GUROBI
start = time.time()
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)
end = time.time()
print("\nSOLVER OK")

print(f"\nTime: {end-start:.2f}s")


print("\n" + "="*40)
print("          ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ          ")
print("="*40)


# Fuel Cell
fc_gas_kw = pyomo.value(model.x_gas_fc) / 1000                          
print(f"Fuel Cell Capacity:     {fc_gas_kw:.2f} kW")

# PV
pv_cap_kw = pyomo.value(model.x_el_pv) / 1000
print(f"PV Capacity:            {pv_cap_kw:.2f} kWp")

# Solar Thermal
st_area = pyomo.value(model.x_th_st)
print(f"Solar Thermal Area:     {st_area:.2f} m²")

# Heat Pump
hp_el_kw = pyomo.value(model.x_el_hp) / 1000                
print(f"Heat Pump Capacity:     {hp_el_kw:.2f} kW")

# Boiler
boiler_kw = pyomo.value(model.x_gas_boiler) / 1000
print(f"Gas Boiler Capacity:    {boiler_kw:.2f} kW")

# Battery
bat_cap_kwh = pyomo.value(model.y_el_battery) / 3.6e6  
print(f"Battery Capacity:       {bat_cap_kwh:.2f} kWh")

# Thermal Tank
tank_cap_kwh = pyomo.value(model.heat_store_capacity) / 3.6e6
print(f"Thermal Tank Capacity:  {tank_cap_kwh:.2f} kWh")
tank_height = pyomo.value(model.y_h_tank)             
tank_radius = model.heat_store_F / 2.0
tank_vol_m3 = 3.14 * (tank_radius**2) * tank_height
tank_vol_liters = tank_vol_m3 * 1000

print("="*40)
print(f"Thermal Tank Height:    {tank_height:.2f} m")
print(f"Thermal Tank Volume:    {tank_vol_liters:.0f} Liters ({tank_vol_m3:.2f} m³)")
print("="*40)
total_cost = pyomo.value(model.objective)
print(f"TOTAL ANNUALIZED COST:          {total_cost:,.2f} CHF/year")
print("="*40)