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
model.T = pyomo.Set(initialize=range(1000)) 
eps = np.finfo(float).eps

# SOLAR RADIATION DATA FROM CSV FILE
data = pd.read_csv("SOLAR_DATA.csv", parse_dates=["datetime"])
data = data.set_index("datetime")
ghi_hourly_values = data["GHI"].values[:8760]

def radiation_initialize(model, t):
    return float(ghi_hourly_values[t])
model.I_t = pyomo.Param(model.T, initialize=radiation_initialize, domain=pyomo.NonNegativeReals)


# ELECTRICITY DEMAND  
electricity_demand = pd.read_csv("RICHARDSON_1.csv", parse_dates=["datetime"])
electricity_demand = electricity_demand.set_index("datetime").resample("h").mean()
assert len(electricity_demand) == 8760
electricity_load = electricity_demand["load_W"].values[:8760]

def L_electricity_initialize(model, t):
    return float(electricity_load[t])
model.L_electricity = pyomo.Param(model.T, initialize=L_electricity_initialize,domain=pyomo.NonNegativeReals)


# DHW DEMAND 
dhw_demand = pd.read_csv("DHW_1.csv", parse_dates=["datetime"])
dhw_demand = dhw_demand.set_index("datetime").resample("h").mean()
dhw_load = dhw_demand["dhw_W"].values[:8760]

def L_dhw_initialize(model, t):
    return float(dhw_load[t])
model.L_dhw = pyomo.Param(model.T, initialize=L_dhw_initialize, domain=pyomo.NonNegativeReals)


# SPACE HEAT DEMAND
sph_demand = pd.read_csv("SPACE_HEAT.csv", parse_dates=["datetime"])
sph_demand = sph_demand.set_index("datetime").resample("h").mean()
sph_load = sph_demand["Q_space_W"].values[:8760]

def L_sph_initialize(model, t):
    return float(sph_load[t])
model.L_sph = pyomo.Param(model.T, initialize=L_sph_initialize, domain=pyomo.NonNegativeReals)


# TEMPERATURE DATA
temperature_demand = pd.read_csv("TEMPERATURES_1.csv", parse_dates=["datetime"])
temperature_demand = temperature_demand.set_index("datetime").resample("h").mean().iloc[:8760]
Tamb_series = temperature_demand["Tamb_C"].values
Tcoll_series = temperature_demand["Tcoll_C"].values

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

# 3.13
model.T_bdg = pyomo.Var(model.T, domain=pyomo.PositiveReals)

# 3.14
model.dT = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

print("DECISION VARIABLES OK")

# Parameters - Table 4 - Table 5

model.Dt = pyomo.Param(initialize=3600)   
model.co2_price = pyomo.Param(initialize=0.06)
 
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
#model.battery_h_plus = pyomo.Param(initialize=0.959)  
#model.battery_h_minus = pyomo.Param(initialize=0.959) 
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

model.y_dhw_in_tank = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)   
model.y_dhw_out_tank = pyomo.Var(model.T, domain=pyomo.NonNegativeReals) 

# 3.14 
model.c_T = pyomo.Param(initialize=1.0)
model.T_min = pyomo.Param(initialize=295) #22°C
model.T_max = pyomo.Param(initialize=297) #24°C

print("PARAMETERS OK")

# EQUATIONS
# Heat Store
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


# Objective Function
# 2.1a
def objective_rule(model):
    # Cx
    fc_capital = model.fc_Ccap * model.fc_h_el * (model.x_gas_fc / 1000)
    pv_capital = model.pv_Ccap * (model.x_el_pv / 1000)
    st_capital = model.st_Ccap * (model.x_th_st / 1000)
    hp_capital = model.hp_Ccap * (model.x_el_hp / 1000)
    boiler_capital = model.boiler_Ccap * model.boiler_h * (model.x_gas_boiler / 1000)
    
    #Cy
    battery_capital = model.battery_Ccap * (model.y_el_battery / 3.6e6)
    tank_capital = model.heat_store_Ccap * (model.heat_store_capacity / 3.6e6)

      
    total_capital = (fc_capital + pv_capital + st_capital + hp_capital + 
                    boiler_capital + battery_capital + tank_capital)
         
    total_operational = 0.0
    
    for t in model.T:
        
        fc_fuel_cost = ((model.fc_Cfuel + model.co2_price * model.fc_pco2) * 
                       model.x_gas_in_fc[t] * model.Dt / 3.6e6)
        
        boiler_fuel_cost = ((model.boiler_Cfuel + model.co2_price * model.boiler_pco2) * 
                           model.x_gas_in_boiler[t] * model.Dt / 3.6e6)
        
        hp_elec_cost = ((model.hp_Cfuel + model.co2_price * model.hp_pco2) * 
                       model.x_el_in_hp[t] * 
                       model.Dt / 3.6e6)
        
        comfort_penalty = model.c_T * model.dT[t]
        
        total_operational += fc_fuel_cost + boiler_fuel_cost + hp_elec_cost + comfort_penalty
    
    return total_capital + total_operational


model.obj = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)


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

# 3.1
def fc_heat_ub(model, t):
        return (model.x_sph_out_fc[t] + model.x_dhw_out_fc[t]) <= model.fc_h_th * model.x_gas_in_fc[t]
model.constraint_fc_heat = pyomo.Constraint(model.T, rule=fc_heat_ub)

# 3.3
def pv_electricity_ub(model, t):
    return model.x_el_out_pv[t] <= (model.pv_h_el * model.pv_h_pr * (model.I_t[t] / 1000.0) * (model.x_el_pv / model.pv_p))
model.constraint_pv_elec = pyomo.Constraint(model.T, rule=pv_electricity_ub)

# 3.4
def st_heat_ub(model, t):
    return (model.x_sph_out_st[t] + model.x_dhw_out_st[t]) <= ((model.st_h * model.I_t[t] - model.st_K * (model.T_coll[t] - model.T_amb[t])) * (model.x_th_st)) / model.st_p
model.constraint_st_heat = pyomo.Constraint(model.T, rule=st_heat_ub)

# 3.6
def hp_space_heat_ub(model, t):
    return model.x_sph_out_hp[t] <= model.hp_h_cop * model.x_el_hp
model.constraint_hp_sph = pyomo.Constraint(model.T, rule=hp_space_heat_ub)

# 3.7 electricity used to power heat pump
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

print("CONVERSION CONSTRAINTS OK")

# 3.10
def battery_charge_ub(model, t):
    return model.y_el_in_battery[t] <= model.battery_p_plus * model.y_el_battery
model.constraint_battery_ch_ub = pyomo.Constraint(model.T, rule=battery_charge_ub)

# 3.10
def battery_discharge_ub(model, t):
    return model.y_el_out_battery[t] <= model.battery_p_minus * model.y_el_battery
model.constraint_battery_dis_ub = pyomo.Constraint(model.T, rule=battery_discharge_ub)

# 3.10
def battery_soc_lb(model, t):

    soc = sum(
        (model.battery_E ** (t - tau)) * (model.battery_h_plus *  model.y_el_in_battery[tau] - model.battery_h_minus * model.y_el_out_battery[tau]) * model.Dt
        for tau in range(min(t + 1, len(model.T)))
    ) + (model.battery_E ** t) * model.battery_yo 
    
    return model.battery_mo * model.y_el_battery <= soc
model.constraint_battery_soc_lb = pyomo.Constraint(model.T, rule=battery_soc_lb)

# 3.10
def battery_soc_ub(model, t):
    soc = sum(
        (model.battery_E ** (t - tau)) * (model.battery_h_plus *  model.y_el_in_battery[tau] - model.battery_h_minus * model.y_el_out_battery[tau]) * model.Dt
        for tau in range(min(t + 1, len(model.T)))
    ) + (model.battery_E ** t) * model.battery_yo 
    
    return soc <= model.y_el_battery
model.constraint_battery_soc_ub = pyomo.Constraint(model.T, rule=battery_soc_ub)

# 3.10
def battery_soc_terminal(model):

    T = len(model.T) - 1 
    soc = sum((model.battery_E ** (T - tau)) * (model.battery_h_plus * model.y_el_in_battery[tau] - model.battery_h_minus * model.y_el_out_battery[tau]) * model.Dt
        for tau in model.T
    ) + (model.battery_E ** T) * model.battery_yo
    
    return soc == model.battery_yT
model.constraint_battery_soc_term = pyomo.Constraint(rule=battery_soc_terminal)
 
# 3.10
def battery_cycling_limit(model):
    T = len(model.T)
    total_discharge = sum(model.y_el_out_battery[t] for t in model.T)
    max_discharge = (T / 24.0) * model.battery_h_minus * (1 - model.battery_mo) * model.y_el_battery

    return total_discharge <= max_discharge
model.constraint_battery_cycling = pyomo.Constraint(rule=battery_cycling_limit)

# 3.12
def tank_soc_lb(model, t):
    soc = sum(
        model.y_dhw_net_tank[tau] - model.heat_store_m1 - model.heat_store_m2
        for tau in range(min(t + 1, len(model.T)))
    ) * model.Dt + model.heat_store_yo  
    
    return eps <= model.heat_store_capacity
model.constraint_tank_soc_lb = pyomo.Constraint(model.T, rule=tank_soc_lb)

# 3.12
def tank_soc_ub(model, t):
    soc = sum(
        model.y_dhw_net_tank[tau] - model.heat_store_m1 - model.heat_store_m2
        for tau in range(min(t + 1, len(model.T)))
    ) * model.Dt + model.heat_store_yo  
    
    return soc <= model.heat_store_capacity
model.constraint_tank_soc_ub = pyomo.Constraint(model.T, rule=tank_soc_ub)

# 3.12
def tank_soc_terminal(model):
    soc_final = sum(
        model.y_dhw_net_tank[tau] - model.heat_store_m1 - model.heat_store_m2
        for tau in model.T
    ) * model.Dt + model.heat_store_yo  
    
    return soc_final == model.heat_store_yT
model.constraint_tank_soc_term = pyomo.Constraint(rule=tank_soc_terminal)


# 3.13
model.U = pyomo.Param(initialize=168.0) #3*56=168 #U≈1 W/K #U=1*168
model.C = pyomo.Param(initialize=3.5e6) 

def tau_initialize(model):
    return model.C / model.U
model.tau = pyomo.Param(initialize=tau_initialize)

def exponential_initialize(model):
    return math.exp((-model.Dt) / model.tau)
model.exponential = pyomo.Param(initialize=exponential_initialize)

def one_minus_exponential_initialize(model):
    return 1.0 - model.exponential
model.one_minus_exponential = pyomo.Param(initialize=one_minus_exponential_initialize)

def Q_plus(model, t):
    return (model.x_sph_out_hp[t] + model.x_sph_out_fc[t] + model.x_sph_out_st[t] + model.x_sph_out_boiler[t])
model.Q_plus = pyomo.Expression(model.T, rule=Q_plus)

def Q_minus(model, t):
    return 0.0
model.Q_minus = pyomo.Expression(model.T, rule=Q_minus)

def building_temperature_rule(model, t):
    if t == 0:
        return model.T_bdg[t] == model.T_amb[t]
    else:
        return (model.T_bdg[t] == 
            model.exponential * model.T_bdg[t - 1] + model.one_minus_exponential * model.T_amb[t] +
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

print("STORAGE CONSTRAINTS OK")


# DEMANDS
# NET FLOW
def dhw_net_flow_rule(model, t):
    return model.y_dhw_net_tank[t] == model.y_dhw_in_tank[t] - model.y_dhw_out_tank[t]
model.constraint_dhw_net = pyomo.Constraint(model.T, rule=dhw_net_flow_rule)

# ELECTRICITY BALANCE
def electricity_balance_rule(model, t):
    supply = (model.x_el_out_fc[t] + model.x_el_out_pv[t] + model.y_el_out_battery[t])
    demand = model.L_electricity[t] + model.x_el_in_hp[t] + model.y_el_in_battery[t]
    return supply == demand
model.constraint_electricity_balance = pyomo.Constraint(model.T, rule=electricity_balance_rule)

# SPACE HEAT BALANCE
def space_heating_balance_rule(model, t):
    supply = (model.x_sph_out_fc[t] + model.x_sph_out_boiler[t] + model.x_sph_out_hp[t] + model.x_sph_out_st[t])
    demand = model.L_sph[t]
    return supply == demand  
model.constraint_space_heating_balance = pyomo.Constraint(model.T, rule=space_heating_balance_rule)

# DHW BALANCE
def dhw_balance_rule(model, t):
    supply = (model.x_dhw_out_fc[t] + model.x_dhw_out_boiler[t] + model.x_dhw_out_st[t])
    demand = model.L_dhw[t]
    return supply + model.y_dhw_out_tank[t] - model.y_dhw_in_tank[t] == demand
model.constraint_dhw_balance = pyomo.Constraint(model.T, rule=dhw_balance_rule)

print("LOAD BALANCE CONSTRAINTS OK")



# GUROBI
print("SOLVING THE MODEL")

solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# RESULTS NOT OK  
print("SOLUTION COMPLETED")
print("Objective Value: ", pyomo.value(model.obj))
print("Fuel Cell: ", pyomo.value(model.x_gas_fc))
print("PV: ", pyomo.value(model.x_el_pv))
print("Solar Thermal: ", pyomo.value(model.x_th_st))
print("Heat Pump: ", pyomo.value(model.x_el_hp))
print("Boiler: ", pyomo.value(model.x_gas_boiler))
print("Battery: ", pyomo.value(model.y_el_battery))
print("Heat Tank: ", pyomo.value(model.y_h_tank))
 