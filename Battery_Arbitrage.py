import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from datetime import date, timedelta
import datetime
import os



# DATA IMPORT

raw_data = pd.read_csv('data/elspotprices_19to21.csv')


#DATA PREPARATION
elspot = raw_data[['HourUTC', 'PriceArea', 'SpotPriceEUR']]
elspot['HourUTC'] = pd.to_datetime(elspot['HourUTC'])
elspot = elspot.pivot(index='HourUTC', columns='PriceArea', values='SpotPriceEUR')

# DAY SELECTION
year    = 2021
month   = 1
day     = 14
price_area = 'DK2'

#%%
# PARAMETERS
P_up    = 1 # MW
SOC_0   = 1 # MWh
SOC_max = 2 # MWh
SOC_min = 0.1 * SOC_max # MWh
T_delta = 1 # hour
eff_cha = 0.9999
eff_dis = 0.9999

#%% OPTIMIZATION

import pyomo.kernel as pyo 

def optimization(year, month, day, price_area):

    price   = elspot[(elspot.index.year == year) &(elspot.index.month == month) &(elspot.index.day == day)][price_area]    

    # MODEL CREATION
    model = pyo.block()
    
    # VARIABLES 
    
    # Power charge
    model.p_cha = pyo.variable_dict()
    for i in range(24):
        model.p_cha[i] = pyo.variable()
        
    # Power discharge
    model.p_dis = pyo.variable_dict()
    for i in range(24):
        model.p_dis[i] = pyo.variable()
        
    # State Of Charge
    model.SOC = pyo.variable_dict()
    for i in range(24):
        model.SOC[i] = pyo.variable()
        
        
    
    # OBJECTIVE FUNCTION
    model.min_cost = pyo.objective( sum(price[t]*(model.p_cha[t] - model.p_dis[t]) for t in range(24)))
    
    
    
    # CONSTRAINTS
    
    # CHARGE Lower bound 
    model.p_cha_lb = pyo.constraint_dict()
    for i in range(24):
        model.p_cha_lb[i] = pyo.constraint( model.p_cha[i] >= 0)
        
    # CHARGE Upper bound 
    model.p_cha_ub = pyo.constraint_dict()
    for i in range(24):
        model.p_cha_ub[i] = pyo.constraint( model.p_cha[i] <= P_up)
    
    # DISCHARGE Lower bound 
    model.p_dis_lb = pyo.constraint_dict()
    for i in range(24):
        model.p_dis_lb[i] = pyo.constraint( model.p_dis[i] >= 0)
        
    # DISCHARGE Upper bound 
    model.p_dis_ub = pyo.constraint_dict()
    for i in range(24):
        model.p_dis_ub[i] = pyo.constraint( model.p_dis[i] <= P_up)
    
    # SOC
    model.SOC_0 = pyo.constraint( model.SOC[0] == SOC_0 )
    model.SOC_23 = pyo.constraint( model.SOC[23] == SOC_0 )
    
    model.SOC_timestep = pyo.constraint_dict()
    for i in range(1, 24):
        model.SOC_timestep[i] = pyo.constraint( model.SOC[i] == model.SOC[i-1] + model.p_cha[i]* eff_cha * T_delta \
                                               -  (model.p_dis[i]/ eff_dis ) * T_delta )
    
    # CAPACITY Lower bound
    model.cap_lb = pyo.constraint_dict()
    for i in range(24):
        model.cap_lb[i] = pyo.constraint( model.SOC[i] >= SOC_min)
        
    # CAPACITY Upper bound
    model.cap_ub = pyo.constraint_dict()
    for i in range(24):
        model.cap_ub[i] = pyo.constraint( model.SOC[i] <= SOC_max)
        
        
    # SOLVER
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee = True)
    
    SOC_result = np.zeros(24)
    dis_result = np.zeros(24)
    cha_result = np.zeros(24)
    profit = 0
    for i in model.SOC:
        SOC_result[i] = model.SOC[i].value
        dis_result[i] = model.p_dis[i].value
        cha_result[i] = model.p_cha[i].value
        profit -= price[i]*(model.p_cha[i].value - model.p_dis[i].value)
        
    df = pd.DataFrame(data = np.array([SOC_result, dis_result, cha_result]).transpose(), columns = ["State of charge", "Discharge", "Charge"])
    
    return df, profit

#%%

# Plot for 14/01/2021

# DAY SELECTION
year    = 2021
month   = 1
day     = 14
price_area = 'DK2'

price = elspot[(elspot.index.year == year) &(elspot.index.month == month) &(elspot.index.day == day)][price_area]    
result, profit = optimization(year, month, day, price_area)

# VISUALIZATION


fig, ax1 = plt.subplots(figsize=(10,5))

ax1.hlines(SOC_min, 0, 23, color = '0.5', linestyle = '--')
ax1.hlines(SOC_max, 0, 23, color = '0.5', linestyle = '--')

ax1.set_xlabel('Hours')
ax1.set_ylabel('State of charge [MWh] ; Charge power [MW]')

y1 = result["Charge"]
y2 = -result["Discharge"]

plt.fill_between(range(24), y1, alpha = 0.4, color = 'tab:green', step = 'pre')
plt.fill_between(range(24), y2, alpha = 0.4, color = 'tab:red', step = 'pre')


ax1.plot(result["State of charge"], color = 'blue', label = 'State of charge')
ax1.step(range(24), y1, color = 'green', label = 'Charging power')
ax1.step(range(24), y2, color = 'red', label = 'Discharging power')

ax1.axhline(y=0, color='k')

ax2 = ax1.twinx()
ax2.set_ylabel('Electricity spot price [€]')
ax2.plot(range(24), price, color = '0.7', label = 'Electricity price')

fig.legend(loc = 'upper right', bbox_to_anchor = (0.9, 0.35))
plt.tight_layout()
plt.show()
plt.savefig('SOC_daily.png', dpi = 800)
#%% 

# Run for every day of 2021
  
start_date = datetime.date(2021, 1, 1)
end_date = datetime.date(2022, 1, 1)
delta = datetime.timedelta(days=1)
#price_area = ['DE', 'NO2', 'SE3', 'SE4', 'DK1', 'DK2']
price_area = ['DK2']
results = []
profits = []
profits_zone = []
date_index = []
means = []

for zone in price_area:
    profits = []
    
    for count, i in enumerate(range((end_date - start_date).days)):
        date_model = start_date + i* delta
    
        year    = date_model.year
        month   = date_model.month
        day     = date_model.day
                
        opt = optimization(year, month, day, zone)
        results.append(opt[0])
        profits.append(opt[1])
        date_index.append(date_model)
    print('The yearly sum profit in ' + zone + ' is ' + str(sum(profits)))
    profits_zone.append(sum(profits))
    
means = elspot.DK2.resample('D').mean()
means = means[(means.index.year == 2021)]

#%%
# DATA VISUALIZATION

# YEARLY PROFIT SINGLE AREA
fig, ax1 = plt.subplots(figsize=(15,5))

ax1.plot(pd.date_range('2021-01-01', periods=365).tolist(), profits, color = 'tab:blue', label = 'Daily profits')

ax1.set_xlabel('Date')
ax1.set_ylabel('Profits [€]')

ax2 = ax1.twinx()
ax2.set_ylabel('Electricity price [€]')
ax2.plot(pd.date_range('2021-01-01', periods=365).tolist(), means, color = 'tab:red', label = 'Mean electricity price')


fig.legend(loc = 'upper left', fontsize = 18, bbox_to_anchor = (0.07, 0.92))

plt.tight_layout()
plt.savefig('yearly_profit_singlearea.png', dpi = 800)
plt.show()
#%%
# YEARLY PROFIT PER AREA
plt.figure(figsize=(6,4))
plt.bar(price_area, profits_zone, width = 0.5, )
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.xlabel('Price area')
plt.ylabel('Yearly profit [€]')
plt.tight_layout()
plt.show()
plt.savefig('yearly_profit_area.png', dpi = 800)
