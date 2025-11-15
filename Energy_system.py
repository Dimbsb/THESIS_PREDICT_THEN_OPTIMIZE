#A linear programming approach to the optimization of residential energy systems
import pyomo.environ as pyomo 
 
# initialize the model
model = pyomo.ConcreteModel()

# Time 
model.T = pyomo.Set(initialize=range(8760)) 

#Decision Variables
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

print("Decision variables initialized successfully")

# Parameters
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