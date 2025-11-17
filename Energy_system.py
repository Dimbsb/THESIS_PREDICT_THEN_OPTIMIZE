#A linear programming approach to the optimization of residential energy systems
import pyomo.environ as pyomo 
import numpy as np
import pandas as pd
from math import pi
 
# initialize the model
model = pyomo.ConcreteModel()
# Time 
model.T = pyomo.Set(initialize=range(8760)) 
eps = np.finfo(float).eps

# Decision Variables
# Cogeneration fuel cell
model.fuel_cell_chemical_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.fuel_gas_flow = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.fuel_space_heat = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.fuel_hot_water = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.fuel_electricity = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  

# Photovoltaic system
model.pv_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.pv_electricity = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Solar thermal system
model.solar_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.solar_space_heat = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.solar_hot_water = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Heat pump
model.heat_pump_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.heat_pump_space_heat = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Boiler
model.boiler_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.boiler_gas_flow = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.boiler_space_heat = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.boiler_hot_water = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Battery storage
model.battery_capacity = pyomo.Var(domain=pyomo.NonNegativeReals)
model.battery_charge_power = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.battery_discharge_power = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# Heat tank storage
model.heat_store_height = pyomo.Var(domain=pyomo.NonNegativeReals)
model.heat_store_net_flow = pyomo.Var(model.T, domain=pyomo.Reals)
model.heat_store_charge_power = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)
model.heat_store_discharge_power = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

print("Decision variables initialized successfully")

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
model.fc_pco2 = pyomo.Param(initialize=0.200)   
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
model.hp_pco2 = pyomo.Param(initialize=0.0915)   

# Boiler Parameters
model.boiler_Cinv = pyomo.Param(initialize=178)  
model.boiler_Comex = pyomo.Param(initialize=14)   
model.boiler_Cfuel = pyomo.Param(initialize=0.1186)   
model.boiler_Tlife = pyomo.Param(initialize=17)   
model.boiler_Ccap = pyomo.Param(initialize=28)   
model.boiler_h = pyomo.Param(initialize=0.95)  
model.boiler_pco2 = pyomo.Param(initialize=0.179)   

# Storage Technologies

# Battery Parameters
model.battery_Cinv = pyomo.Param(initialize=1000)   
model.battery_Tlife = pyomo.Param(initialize=9)   
model.battery_Ccap = pyomo.Param(initialize=128)   
model.battery_h = pyomo.Param(initialize=0.92)   
model.battery_E = pyomo.Param(initialize=0.05)  
model.battery_mo = pyomo.Param(initialize=0.20)   
model.battery_p_plus = pyomo.Param(initialize=0.58)   
model.battery_p_minus = pyomo.Param(initialize=1.00)   
model.battery_yo = pyomo.Param(initialize=1.00)  
model.battery_yT = pyomo.Param(initialize=1.00)  
model.battery_h_plus = pyomo.Param(initialize=0.959)  
model.battery_h_minus = pyomo.Param(initialize=0.959) 

# Heat Store Parameters
model.heat_store_Cinv = pyomo.Param(initialize=86)  
model.heat_store_Tlife = pyomo.Param(initialize=20)  
model.heat_store_Ccap = pyomo.Param(initialize=6)  
model.heat_store_F = pyomo.Param(initialize=0.80)  
model.heat_store_U = pyomo.Param(initialize=0.21)  
model.heat_store_Cp = pyomo.Param(initialize=4.19)   
model.heat_store_p = pyomo.Param(initialize=1000)   
model.heat_store_Thot = pyomo.Param(initialize=60)   
model.heat_store_Tcold = pyomo.Param(initialize=10)   
model.heat_store_Tbdg = pyomo.Param(initialize=23)   
model.heat_store_yo = pyomo.Param(initialize=0.50)   
model.heat_store_yT = pyomo.Param(initialize=0.50)   
model.heat_store_mo = pyomo.Param(initialize=0.50)   

# Time parameter
model.Dt = pyomo.Param(initialize=3600)   
model.co2_price = pyomo.Param(initialize=0.06)   

print("Parameters initialized successfully")

# Objective Function

def objective_rule(model):
    # Cx
    conversion_capital = (
        model.fc_Ccap * model.fc_h_el * model.fuel_cell_chemical_capacity +
        model.pv_Ccap * model.pv_capacity +
        model.st_Ccap * model.solar_capacity +
        model.hp_Ccap * model.heat_pump_capacity +
        model.boiler_Ccap * model.boiler_h * model.boiler_capacity
    )
       
    #y
    heat_store_capacity = (
        model.heat_store_p * (model.heat_store_Cp / 3600) * 
        (model.heat_store_Thot - model.heat_store_Tcold) * pi * 
        ((model.heat_store_F / 2)**2) * model.heat_store_height)
    
    
    heat_store_capital = model.heat_store_Ccap * heat_store_capacity
    battery_capital = model.battery_Ccap * model.battery_capacity
    
    # Cy
    storage_capital = battery_capital + heat_store_capital
    
    # Sum
    total_cop = 0.0
    for t in model.T:

        # 3.2
        fuel_cell_cop = ((model.fc_Cfuel + model.co2_price * model.fc_pco2) * model.fuel_gas_flow[t])

        # 3.9
        boiler_cop = ((model.boiler_Cfuel + model.co2_price * model.boiler_pco2) * model.boiler_gas_flow[t])

        # 3.7
        heat_pump_cop = ((model.hp_Cfuel + model.co2_price * model.hp_pco2) * 
                        (model.heat_pump_space_heat[t] / model.hp_h_cop))

        total_cop += fuel_cell_cop + boiler_cop + heat_pump_cop

    # objective function 2.1a
    return conversion_capital + storage_capital + total_cop

model.objective = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)



# Net Flow 
def heat_store_net_flow_def_rule(model, t):
    return model.heat_store_net_flow[t] == (model.heat_store_charge_power[t] - model.heat_store_discharge_power[t])
model.heat_store_net_flow_def = pyomo.Constraint(model.T,rule=heat_store_net_flow_def_rule)



# CONSTRAINT 2.1b 

def fuel_cell_min_threshold(model, t):
    return model.fc_k * model.fuel_cell_chemical_capacity <= model.fuel_gas_flow[t]
model.fuel_cell_min = pyomo.Constraint(model.T, rule=fuel_cell_min_threshold)

def fuel_cell_max_threshold(model, t):
    return model.fuel_gas_flow[t] <= model.fuel_cell_chemical_capacity
model.fuel_cell_max = pyomo.Constraint(model.T, rule=fuel_cell_max_threshold)

def boiler_max_threshold(model, t):
    return model.boiler_gas_flow[t] <= model.boiler_capacity
model.boiler_max = pyomo.Constraint(model.T, rule=boiler_max_threshold)

def heat_pump_max_threshold(model, t):
    return model.heat_pump_space_heat[t] <= model.heat_pump_capacity
model.heat_pump_max = pyomo.Constraint(model.T, rule=heat_pump_max_threshold)


# CONSTRAINT 2.1c

def fuel_cell_electricity_from_fuel_ub(model, t):
    return model.fuel_electricity[t] <= model.fc_h_el * model.fuel_gas_flow[t]
model.fuel_cell_elec_from_fuel_ub = pyomo.Constraint(model.T, rule=fuel_cell_electricity_from_fuel_ub)

def fuel_cell_heat_from_fuel_ub(model, t):
    return (model.fuel_space_heat[t] + model.fuel_hot_water[t]) <= model.fc_h_th * model.fuel_gas_flow[t]
model.fuel_cell_heat_from_fuel_ub = pyomo.Constraint(model.T, rule=fuel_cell_heat_from_fuel_ub)

def boiler_heat_from_gas_ub(model, t):
    return (model.boiler_space_heat[t] + model.boiler_hot_water[t]) <= model.boiler_h * model.boiler_gas_flow[t]
model.boiler_heat_from_gas_ub = pyomo.Constraint(model.T, rule=boiler_heat_from_gas_ub)

def solar_thermal_from_capacity_ub(model, t):
    return (model.solar_space_heat[t] + model.solar_hot_water[t]) <= model.st_h * model.solar_capacity
model.solar_from_capacity_ub = pyomo.Constraint(model.T, rule=solar_thermal_from_capacity_ub)

# SOLAR RADIATION DATA FROM CSV FILE
df2017 = pd.read_csv("2017.csv")
df2017["datetime"] = pd.to_datetime(df2017[["Year", "Month", "Day", "Hour", "Minute"]])
df2017 = df2017.set_index("datetime")
ghi_hourly = df2017["GHI"].resample("h").mean()
assert len(ghi_hourly) == 8760
ghi_hourly_values = ghi_hourly.values
def radiation_init(model, t):
    return float(ghi_hourly_values[t])

model.radiation = pyomo.Param(model.T, initialize=radiation_init, domain=pyomo.NonNegativeReals)

# PV AND HEAT PUMP 2.1b and 2.1c from 3.3 , 3.6 , 3.7
def pv_solar_electricity_rule(model, t):
    return model.pv_electricity[t] <= (model.pv_h_el * model.pv_h_pr * (model.radiation[t] / 1000.0) * (model.pv_capacity/model.pv_p))
model.pv_solar_electricity = pyomo.Constraint(model.T, rule=pv_solar_electricity_rule)

def heat_pump_space_heat_rule(model, t):
    return model.heat_pump_space_heat[t] <= model.hp_h_cop * model.heat_pump_capacity
model.hp_space_heat = pyomo.Constraint(model.T, rule=heat_pump_space_heat_rule)


# CONSTRAINT 2.1d

# CONSTRAINT 2.1e and 2.1f
# Battery

def battery_charge_power_limit_rule(model, t):
    return model.battery_charge_power[t] <= model.battery_p_plus * model.battery_capacity
model.battery_charge_power_limit = pyomo.Constraint(model.T, rule=battery_charge_power_limit_rule)

def battery_discharge_power_limit_rule(model, t):
    return model.battery_discharge_power[t] <= model.battery_p_minus * model.battery_capacity
model.battery_discharge_power_limit = pyomo.Constraint(model.T, rule=battery_discharge_power_limit_rule)

# Heat Store
