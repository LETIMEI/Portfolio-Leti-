# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:41:19 2024

@author:
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import pandas as pd
import numpy as np

file_name = "MSBA-Term---Workshop-and-Seminar-Allocations-1.xlsx"

df = pd.read_excel(file_name, "Sheet2")
df2 = pd.read_excel(file_name, "Sheet3")

students = df.iloc[1:, 0] #focus on MSBA student only
classes = df.iloc[0, 2:] #view different workshop slot as independent classes
time = df2.iloc[0:, 0]

#set a M number of groups
M = 3

df_np = df.to_numpy()
df2_np = df2.to_numpy()

# data preparation

## Ak set: containing students enrolled in class k
participation = df_np[1:, 2:]

## Ct set: containing classes taking place on time t
seminar_time = df2_np[0:, 3:]

## room capacity data
room_cap = df_np[0, 2:]


## ck: social distancing limitatino for class k (assume half of current room cap)
ck = np.floor(room_cap*0.2)

# E: escess room capacity (assumption)
E = 30

# model building
model = ConcreteModel()

#decision variable
model.pi = Var(range(len(students)), range(M), domain = Binary)

model.e = Var(range(M), range(len(classes)), domain = NonNegativeReals)
model.stj = Var(range(len(time)), range(M), domain = Reals)
model.delta = Var(range(M), range(len(classes)), domain = NonNegativeReals)
model.s = Var(domain = NonNegativeReals)


# objective function
model.obj = Objective(expr = sum(model.e[j,k] for j in range(M) for k in range(len(classes))) +\
                             0.5*sum(model.delta[j,k] for j in range(M) for k in range(len(classes))) +\
                                 2*model.s, sense = minimize)
    
#constraints

# constraint(7)
def rule1(model, i):
    return sum(model.pi[i,j] for j in range(M)) == 1

model.constr1 = Constraint(range(len(students)), rule = rule1)

# constraint(8)
def rule2(model, j, k):
    return sum(participation[i,k]*model.pi[i,j] for i in range(len(students)))-ck[k] <= model.e[j,k]

model.constr2 = Constraint(range(M), range(len(classes)), rule = rule2)

# constraint(9-1)
def rule3(model,j,k):
    return -model.delta[j,k] <= sum(
        participation[i,k]*model.pi[i,j] for i in range(len(students))) - \
        sum(participation[i,k] for i in range(len(students)))/M

model.constr3 = Constraint(range(M), range(len(classes)), rule = rule3)

# constraint(9-2)
def rule4(model,j,k):
    return model.delta[j,k] >= sum(
        participation[i,k]*model.pi[i,j] for i in range(len(students))) - \
        sum(participation[i,k] for i in range(len(students)))/M

model.constr4 = Constraint(range(M), range(len(classes)), rule = rule4)

# constraint(10)
def rule5(model, t, j):
    return model.s >= model.stj[t, j] - E

model.constr5 = Constraint(range(len(time)), range(M), rule=rule5)

# constraint(11)
def rule6(model, t, j):
    return model.stj[t,j] == sum(model.e[j,k]*seminar_time[t,k] for k in range(len(classes)))

model.constr6 = Constraint(range(len(time)), range(M), rule=rule6)

# specify the solver we are using and solve the problem
solver = SolverFactory('gurobi')
results = solver.solve(model, tee = True)

# print the objective value and the value of TD, TE, SSE
if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Objective value=", round(model.obj(),2),'\n'
          "TE =", sum(model.e[j,k] for j in range(M) for k in range(len(classes)))(),'\n'
          "TD =", round(sum(model.delta[j,k] for j in range(M) for k in range(len(classes)))(),2),'\n'
          "SSE =", model.s())
else:
    print("Solve Failed")

# print the delta value for each group-class combination
for k in range(len(classes)):
    for j in range(M):
        print("For group", j+1, "in class", k+1, "delta =", round(model.delta[j,k](),2))
 
# print a list showing which group each student is in        
for i in range(len(students)):
    for j in range(M):
        if model.pi[i,j]() > 0.5:
            print("Student", i+1, "is in group", j+1)

# print the stj value for each time-class combination              
for t in range(len(time)):
    for j in range(M):
        print('at time', time[t], model.stj[t,j]())

# print the ejk value (excess for class k) for each group-class combination
for k in range(len(classes)):
    for j in range(M):
        print("If group", j+1, "is assigned to attend class", k+1, "in person,", "excess =", model.e[j,k]())

# print the value of SE (max stj)
max_value = 0
for t in range(len(time)):
    for j in range(M): 
        max_value = max(max_value, model.stj[t, j]())

print("SE =", max_value)


