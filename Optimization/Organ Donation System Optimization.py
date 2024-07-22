# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:38:26 2024

@author: 
"""

# imports pandas package and names it pd
import pandas as pd

# imports numpy package and names it np
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *  # with this we dont need to use pyo to specify

from sympy import symbols, Eq, solve

from sklearn.linear_model import LinearRegression

model = ConcreteModel()

df = pd.read_excel("PatientData.xlsx")

patient = df.loc[df.index[2:], df.columns[1:]].keys()
organ = df.loc[df.index[2:], df.columns[1]].keys()

df_np = df.to_numpy()

LYFT = df_np[2:, 1:]

DT = df_np[0, 1:]

CPRA = df_np[1, 1:]

LowerBound = [3,2,1]

model.x = Var(range(len(organ)), range(len(patient)), bounds=(0,1))


# creates the objective function rule
def objective_rule(model):
    return sum(LYFT[i,j]*model.x[i,j] for i in range(len(organ)) 
               for j in range(len(patient)))

# adds the objectuve function using the rule
model.obj = Objective(rule=objective_rule, sense=maximize)


# creates a constraint rule for each patient p
def patient_rule(model, j):
    return (sum(model.x[i,j] for i in range(len(organ))) <= 1)

# adds the constraint function using the rule
model.PatientRule = Constraint(range(len(patient)), rule = patient_rule)


# creates a constraint rule for each organ o
def organ_rule(model, i):
    return (sum(model.x[i,j] for j in range(len(patient))) <= 1)

# adds the constraint function using the rule
model.OrganRule = Constraint(range(len(organ)), rule = organ_rule)

# lower bound rules for each group
model.Group_1 = Constraint(expr = sum(model.x[i,j] for i in range(len(organ)) for j in range(len(patient)) if DT[j] <= 5) >= 3)

model.Group_2 = Constraint(expr = sum(model.x[i,j] for i in range(len(organ)) for j in range(len(patient)) if DT[j] >= 5 and DT[j] <= 10) >= 2)

model.Group_3 = Constraint(expr = sum(model.x[i,j] for i in range(len(organ)) for j in range(len(patient)) if DT[j] >= 10) >= 1)

model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)  # before solve


solver = SolverFactory('glpk')
results = solver.solve(model, tee = False)

if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Total enlongated year is", model.obj(), "years")
else:
    print("Solve Failed")
    
    
for i in range(len(organ)):
    for j in range(len(patient)):
        if model.x[i,j]() > 0.5:
            print("organ", i+1, "is assigned to patient", j+1)

dval1 = pyo.value(model.dual[model.Group_1])  # after solve
dval2 = pyo.value(model.dual[model.Group_2]) 
dval3 = pyo.value(model.dual[model.Group_3]) 

print("shadow price 1 =", dval1)
print("shadow price 2 =", dval2) 
print("shadow price 3 =", dval3)

#--------------------------------------------------------------------------
# Calculate V(p,o) values
calculated_values = np.zeros((len(organ), len(patient)))

for i in range(len(organ)):
    for j in range(len(patient)):
        value = None
        if DT[j] < 5:
            value = LYFT[i, j] - dval1
        elif DT[j] == 5:
            value = LYFT[i, j] - dval1 - dval2
        elif DT[j] > 5 and DT[j] < 10:
            value = LYFT[i, j] - dval2
        elif DT[j] == 10:
            value = LYFT[i, j] - dval2 - dval3
        elif DT[j] > 10:
            value = LYFT[i, j] - dval3
        calculated_values[i, j] = value

#--------------------------------------------------------------------------
# Another way to get the v(p,o) value
calculated_values = np.zeros((len(organ), len(patient)))
calculated_values = pd.DataFrame(index=len(organ),columns=len(patient))
for i in range(len(organ)):
    for j in range(len(patient)):
        calculated_values[i,j] = LYFT[i, j]
        if DT[j] <= 5:
            calculated_values[i,j] = calculated_values[i,j] - dval1
        if DT[j] >= 5 and DT[j] <= 10:
            calculated_values[i,j] = calculated_values[i,j] - dval2
        if DT[j] >= 10:
            calculated_values[i,j] = calculated_values[i,j] - dval3
#--------------------------------------------------------------------------
# Solve the answer for w0, and coefficients
# Correct answer below
X = np.zeros((len(patient)*len(organ),5))
y = np.zeros(len(patient)*len(organ))
index = 0;
for p in range(len(patient)):
    for o in range(len(organ)):
        X[index,0] = LYFT[o,p]
        X[index,1] = min(DT[p],5)
        X[index,2] = min(max(DT[p]-5,0),5)
        X[index,3] = max(DT[p]-10,0)
        X[index,4] = CPRA[p]
        y[index] = calculated_values[o,p]
        index = index + 1
reg = LinearRegression().fit(X, y)
rsquared = reg.score(X,y)
w0 = reg.intercept_
w = reg.coef_

#--------------------------------------------------------------------------
# Wrong answer here
w0, w1, w2, w3, w4, w5 = symbols('w0 w1 w2 w3 w4 w5')

equations = []

for i in range(len(organ)):
    for j in range(len(patient)):
        if DT[j] < 5:
            equation = Eq(w0+w1*LYFT[i,j]+w2*DT[j]+w5*CPRA[j], calculated_values[i,j])
        if DT[j] == 5:
            equation = Eq(w0+w1*LYFT[i,j]+w2*DT[j]+w3*DT[j]+w5*CPRA[j], calculated_values[i,j])
        if DT[j] > 5 and DT[j] < 10:
            equation = Eq(w0+w1*LYFT[i,j]+w3*DT[j]+w5*CPRA[j], calculated_values[i,j])
        if DT[j] == 10:
            equation = Eq(w0+w1*LYFT[i,j]+w3*DT[j]+w4*DT[j]+w5*CPRA[j], calculated_values[i,j])
        if DT[j] > 10:
            equation =  Eq(w0+w1*LYFT[i,j]+w4*DT[j]+w5*CPRA[j],calculated_values[i,j])
        equations.append(equation)
    
        
solution = solve((equations), (w0, w1, w2, w3, w4, w5))
print("Solution:", solution)
    
for eq in equations:
    print(eq)

# find approximated solution
from sympy import lambdify
from scipy.optimize import minimize

# Convert the symbolic equations to numerical functions
numerical_equations = [lambdify((w0, w1, w2, w3, w4, w5), eq.rhs - eq.lhs) for eq in equations]

initial_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Define a new objective function using the numerical equations
def objective_function(variables):
    values = [eq(*variables) for eq in numerical_equations]
    return np.sum(np.square(values))

# Continue with the optimization as before
result = minimize(objective_function, initial_guess)
print(result)

numerical_solution = result.x
print("Numerical Solution:", numerical_solution)



