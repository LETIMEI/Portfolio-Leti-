# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:53:40 2024

@author: Leti Mei
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import pandas as pd
import numpy as np

file_name = "shsm-n50-t50-cap100.xlsx"

df = pd.read_excel(file_name, "VM")
df2 = pd.read_excel(file_name, "PM")

# Determine the time horizon
time_horizon = df['end'].max()

# Initialize the a_vt matrix
a_vt = np.zeros((len(df), time_horizon + 1))

# Populate the a_vt matrix
for index, row in df.iterrows():
    vm = row['VM']
    start = row['start']
    end = row['end']
    cpu_request = row['CPU_request']
    a_vt[vm, start:end + 1] = cpu_request
    
    
# Convert to DataFrame for better readability
a_vt_df = pd.DataFrame(a_vt, columns=[f't{t}' for t in range(time_horizon + 1)])
a_vt_df.index = [f'v{v}' for v in range(len(df))]

print(a_vt_df)

# model building
model = ConcreteModel()


# Define sets
model.V = pyo.Set(initialize=df['VM'].tolist())
model.T = pyo.Set(initialize=range(1, time_horizon + 1))
model.P = pyo.Set(initialize=df2['PM'].tolist())

# Define parameters
model.a_vt = pyo.Param(model.V, model.T, initialize=lambda model, v, t: a_vt_df.loc[f'v{v}', f't{t}'])
model.capacity = pyo.Param(model.P, initialize=dict(zip(df2['PM'], df2['capacity'])))


# Variables
model.x_vp = pyo.Var(model.V, model.P, within=pyo.Binary)  # 1 if VM v is placed on PM p
model.y_pt = pyo.Var(model.P, model.T, within=pyo.Binary)  # 1 if PM p is used at time t
model.z_p = pyo.Var(model.P, within=pyo.Binary)  # 1 if PM p is used

# Objective: Minimize the number of physical machines used
def objective_function(model):
    return pyo.summation(model.z_p)

model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Constraints
# Each VM must be placed on exactly one PM
def vm_placement_constraint(model, v):
    return sum(model.x_vp[v, p] for p in model.P) == 1

model.vm_placement_constraint = pyo.Constraint(model.V, rule=vm_placement_constraint)

# Capacity constraints for each PM at each time t
def capacity_constraint(model, p, t):
    return sum(model.a_vt[v, t] * model.x_vp[v, p] for v in model.V if (v, t) in model.a_vt) <= model.capacity[p] * model.y_pt[p, t]

model.capacity_constraint = pyo.Constraint(model.P, model.T, rule=capacity_constraint)


# Linking z_p with y_pt
def z_p_constraint(model, p, t):
    return model.z_p[p] >= model.y_pt[p, t]

model.z_p_constraint = pyo.Constraint(model.P, model.T, rule=z_p_constraint)

# specify the solver we are using and solve the problem
solver = SolverFactory('gurobi')
results = solver.solve(model, tee = True)

# print the objective value and the value of TD, TE, SSE
if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Objective value=", round(model.objective(),2))
else:
    print("Solve Failed")
    
    
# Print the values of x_vp
print("\nVM Placement (x_vp):")
for v in model.V:
    for p in model.P:
        if pyo.value(model.x_vp[v, p]) == 1:
            print(f"VM {v} is placed on PM {p}")
    
    
#-------------------------------------------------------------------------------
def first_fit_decreasing_heuristic3(vm_data, pm_data, time_horizon):
    # Get the list of PMs and their capacities
    pms = pm_data['PM'].tolist()
    capacities = pm_data['capacity'].tolist()

    # Initialize placement and remaining capacities for each time slot
    vm_placement = {pm: [] for pm in pms}
    remaining_capacity = {pm: [capacity] * (time_horizon + 1) for pm, capacity in zip(pms, capacities)}
    
    # Sort VMs in descending order based on CPU request
    sorted_vms = vm_data.sort_values(by='CPU_request', ascending=False)
    
    for _, vm in sorted_vms.iterrows():
        vm_id = vm['VM']
        vm_start = vm['start']
        vm_end = vm['end']
        vm_cpu = vm['CPU_request']
        
        # Find the first PM that can fit the VM
        placed = False
        for pm in pms:
            if all(remaining_capacity[pm][t] >= vm_cpu for t in range(vm_start, vm_end + 1)):
                # Place VM on this PM
                vm_placement[pm].append(vm_id)
                
                # Update remaining capacity
                for t in range(vm_start, vm_end + 1):
                    remaining_capacity[pm][t] -= vm_cpu
                
                placed = True
                break
        
        if not placed:
            print(f"VM {vm_id} could not be placed.")
    
    # Remove empty PMs from the placement
    vm_placement = {pm: vms for pm, vms in vm_placement.items() if vms}
    
    return vm_placement

# Apply the First Fit Decreasing Heuristic
vm_placement = first_fit_decreasing_heuristic3(df, df2, time_horizon)

# Print VM Placement
for pm, vms in vm_placement.items():
    print(f"PM {pm} has VMs: {vms}")


#-------------------------------------------------------------------------------
# Best Fit Decreasing Heuristic for VM Placement
def best_fit_decreasing_heuristic(vm_data, pm_data, time_horizon):
    # Initialize placement and remaining capacities for each time slot
    vm_placement = {}
    remaining_capacity = {pm: [pm_data['capacity'][0]] * (time_horizon + 1) for pm in pm_data['PM']}
    
    # Sort VMs in descending order based on CPU request
    sorted_vms = vm_data.sort_values(by='CPU_request', ascending=False)
    
    for vm_index, vm in sorted_vms.iterrows():
        vm_id = vm['VM']
        vm_start = vm['start']
        vm_end = vm['end']
        vm_cpu = vm['CPU_request']
        
        # Find the best PM that can fit the VM
        best_fit_pm = None
        min_wasted_capacity = float('inf')
        
        for pm in pm_data['PM']:
            if all(remaining_capacity[pm][t] >= vm_cpu for t in range(vm_start, vm_end + 1)):
                # Calculate the wasted capacity if this VM is placed on this PM
                wasted_capacity = sum(remaining_capacity[pm][t] - vm_cpu for t in range(vm_start, vm_end + 1))
                
                # Check if this PM has less wasted capacity than the current best fit
                if wasted_capacity < min_wasted_capacity:
                    best_fit_pm = pm
                    min_wasted_capacity = wasted_capacity
        
        if best_fit_pm is not None:
            # Place VM on the best fit PM
            if best_fit_pm not in vm_placement:
                vm_placement[best_fit_pm] = []
            vm_placement[best_fit_pm].append(vm_id)
            
            # Update remaining capacity
            for t in range(vm_start, vm_end + 1):
                remaining_capacity[best_fit_pm][t] -= vm_cpu
        else:
            print(f"VM {vm_id} could not be placed.")
    
    return vm_placement

# Apply the Best Fit Decreasing Heuristic
vm_placement = best_fit_decreasing_heuristic(df, df2, time_horizon)

# Print VM Placement
for pm, vms in vm_placement.items():
    print(f"PM {pm} has VMs: {vms}")






