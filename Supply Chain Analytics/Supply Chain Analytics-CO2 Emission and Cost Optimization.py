# -*- coding: utf-8 -*-
"""
Minimizing cost

Data:
    di: the distance between ODM i and the distribution centre
    cxi: the production cost of LCD42 at ODM i
    cyi: the production cost of LCD32 at ODM i
    w: product weight in metric ton
    t: units of each product to be shipped
    sij: shipping cost from ODM i via transportation j
    
    
Decision Variable:
    xij: the amount of LCD42 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    yij: the amount of LCD32 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    ai: whether ODM i is selected to produce LCD42
    bi: whether ODM i is selected to produce LCD32
    

@author: 
"""

# import packages
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import pandas as pd
import numpy as np

# list data given in the question
d = [2508, 1553, 1380, 2150, 30, 690, 686] #distance
cx = [1983.4, 2254, 2582.4, 1976.1, 2711.3, 2704.8, 2125.2] #production cost of LCD42
cy = [1818, 1996.4] #production cost of LCD32
w = [0.022, 0.0165] #product weight in metric ton
t = [920000, 530000] #target shipping unit
s = [64400, 70840, 6182.4, 5216.4, 4830, 4250.4, 3091.2,
     115920, 127512, 7084, 5796, 5667.2, 5796, 2704.8,
     103040, 113344, 7084, 5796, 5667.2, 5796, 3284.4,
     64400, 70840, 6182.4, 5280.8, 5216.4, 4250.4, 3091.2,
     0, 0, 9660, 9016, 8694, 0, 0,
     135240, 148120, 0, 0, 0, 0, 3413.2,
     103040, 112700, 7084, 5796, 5538.4, 5860.4, 2729.2] #shipping cost
emission = [1.44, 1.44, 0.0613, 0.0613, 0.0613, 0.0285, 0.007]



ODM = ("ODM1", "ODM2", "ODM3", "ODM4", "ODM5", "ODM6", "ODM7")
Transportation = ("Regular Air", "Air Express", "Road", "Road LTL", "Road Network", "Rail", "Water")


# turning shipping cost list into a 7*7 table
s = np.array(s).reshape(7, 7)

# create a blanck concrete model
model = ConcreteModel()

# create variable xij for the amount of LCD42 produced by ODM i shipped via transportation j
model.x = Var(range(len(ODM)), range(len(Transportation)), domain=NonNegativeReals)

# creates variable yij for the amount of LCD32 produced by ODM i shipped via transportation j
model.y = Var(range(len(ODM[0:2])), range(len(Transportation)), domain=NonNegativeReals)

# create binary variable ai for whether ODM i is chosen for producing LCD42
model.a = Var(range(len(ODM)), within=(0,1))

# create binary variable bi for whether ODM i is chosen for producing LCD32
model.b = Var(range(len(ODM[0:2])), within=(0,1))


# creates the objective function rule
def objective_rule(model): # for shipping cost
    return sum(w[0]*s[i,j]*model.x[i,j] for i in range(len(ODM)) 
               for j in range(len(Transportation))) + \
        sum(w[1]*s[k, j]*model.y[k,j] for k in range(len(ODM[0:2])) 
            for j in range(len(Transportation))) + \
        sum(cx[i]*sum(model.x[i, j] for j in range(len(Transportation))) for i in range(len(ODM))) + \
        sum(cy[k]*sum(model.y[k, j] for j in range(len(Transportation))) for k in range(len(ODM[0:2])))

# adds the objectuve function using the rule
model.obj = Objective(rule=objective_rule, sense=minimize)


# Shipping target constraint
model.target42 = Constraint(expr = sum(model.x[i,j] for i in range(len(ODM)) 
                                       for j in range(len(Transportation))) >= t[0])

model.target32 = Constraint(expr = sum(model.y[k,j] for k in range(len(ODM[0:2])) 
                                       for j in range(len(Transportation))) >= t[1])

# budget constraint
model.budget = Constraint(expr = sum(w[0]*s[i,j]*model.x[i,j] for i in range(len(ODM)) 
           for j in range(len(Transportation))) + \
    sum(w[1]*s[k, j]*model.y[k,j] for k in range(len(ODM[0:2])) 
        for j in range(len(Transportation))) + \
    sum(cx[i]*sum(model.x[i, j] for j in range(len(Transportation))) for i in range(len(ODM))) + \
    sum(cy[k]*sum(model.y[k, j] for j in range(len(Transportation))) for k in range(len(ODM[0:2]))) <= 3000000000)

# Production constraints (upper bound)
model.upperbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.upperbound.add(sum(model.y[k,j] for j in range(len(Transportation))) <= 600000 * model.b[k])

    
for i in range(len(ODM)):
    model.upperbound.add(sum(model.x[i,j] for j in range(len(Transportation))) <= 600000*model.a[i])


# Production constraints (lower bound)
model.lowerbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.a[k])

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.b[k])

for i in range(2,7):
    model.lowerbound.add(sum(model.x[i,j] for j in range(len(Transportation))) >= 200000*model.a[i])


# Minimum number of products to be shipped via Regular Air or Air Express      
model.air42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(len(Transportation[0:2]))) >= 46000)

model.air32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(len(Transportation[0:2]))) >= 53000)


# Minimum number of products to be shipped via Road or Road LTL or Road Network
model.road42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(2,5)) >= 92000)

model.road32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(2,5)) >= 79500)


# Minimum number of products to be shipped via Rail
model.rail42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(5,6)) >= 138000)

model.rail32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(5,6)) >= 79500)


# Infeasible transportation method
model.infeasible = ConstraintList()
model.infeasible.add(model.x[4,0] == 0)
model.infeasible.add(model.x[4,1] == 0)
model.infeasible.add(model.x[4,5] == 0)
model.infeasible.add(model.x[4,6] == 0)

model.infeasible.add(model.x[5,2] == 0)
model.infeasible.add(model.x[5,3] == 0)
model.infeasible.add(model.x[5,4] == 0)
model.infeasible.add(model.x[5,5] == 0)


solver = SolverFactory('glpk')
results = solver.solve(model,tee=False)

if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Total cost is", round(model.obj(),2))
else:
    print("Solve Failed")
    
for i in range(len(ODM)):
    print("ODM", i+1, "should produce", round(sum(model.x[i,j] for j in range(len(Transportation)))(),2), "units of LCD42"
          )

for i in range(len(ODM)):
    for j in range(len(Transportation)):
            print("ODM", i+1, "should ship", model.x[i,j](),"LCD42 via", Transportation[j])

for k in range(len(ODM[0:2])):
    print("ODM", k+1, "should produce", sum(model.y[k,j] for j in range(len(Transportation)))(), "units of LCD32"
          )

for k in range(len(ODM[0:2])):
    for j in range(len(Transportation)):
            print("ODM", k+1, "should ship", model.y[k,j](),"LCD32 via", Transportation[j])


for i in range(len(ODM)):
    if model.a[i]() > 0.5:
        print("ODM", i+1, "is selected to produce LCD42")


for k in range(len(ODM[0:2])):
    if model.b[k]() > 0.5:
        print("ODM", k+1, "is selected to produce LCD32")


if results.solver.termination_condition == TerminationCondition.optimal:
    # Calculate the CO2 emission
    co2_emission = sum(d[i]*w[0]*model.x[i,j]*emission[j] for i in range(len(ODM)) for j in range(len(Transportation))) + \
        sum(d[k]*w[1]*model.y[k,j]*emission[j] for k in range(len(ODM[0:2])) for j in range(len(Transportation)))
    print("CO2 emission:", co2_emission())




"""
Minimizing CO2 emission under 3 billion budget

Data:
    di: the distance between ODM i and the distribution centre
    cxi: the production cost of LCD42 at ODM i
    cyi: the production cost of LCD32 at ODM i
    w: product weight in metric ton
    t: units of each product to be shipped
    sij: shipping cost from ODM i via transportation j
    emission,j: CO2 emission in kg for transportation j per ton-km shipped
    
Decision Variable:
    xij: the amount of LCD42 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    yij: the amount of LCD32 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    ai: whether ODM i is selected to produce LCD42
    bi: whether ODM i is selected to produce LCD32
    
"""
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import pandas as pd
import numpy as np

d = [2508, 1553, 1380, 2150, 30, 690, 686]
cx = [1983.4, 2254, 2582.4, 1976.1, 2711.3, 2704.8, 2125.2]
cy = [1818, 1996.4]
w = [0.022, 0.0165]
t = [920000, 530000]
s = [64400, 70840, 6182.4, 5216.4, 4830, 4250.4, 3091.2,
     115920, 127512, 7084, 5796, 5667.2, 5796, 2704.8,
     103040, 113344, 7084, 5796, 5667.2, 5796, 3284.4,
     64400, 70840, 6182.4, 5280.8, 5216.4, 4250.4, 3091.2,
     0, 0, 9660, 9016, 8694, 0, 0,
     135240, 148120, 0, 0, 0, 0, 3413.2,
     103040, 112700, 7084, 5796, 5538.4, 5860.4, 2729.2]
emission = [1.44, 1.44, 0.0613, 0.0613, 0.0613, 0.0285, 0.007]


ODM = ("ODM1", "ODM2", "ODM3", "ODM4", "ODM5", "ODM6", "ODM7")
Transportation = ("Regular Air", "Air Express", "Road", "Road LTL", "Road Network", "Rail", "Water")


# turning shipping cost list into a 7*7 table
s = np.array(s).reshape(7, 7)

# create a blanck concrete model
model = ConcreteModel()

# create variable xij for the amount of LCD42 produced by ODM i shipped via transportation j
model.x = Var(range(len(ODM)), range(len(Transportation)), domain=NonNegativeReals)
 
# creates variable yij for the amount of LCD32 produced by ODM i shipped via transportation j
model.y = Var(range(len(ODM[0:2])), range(len(Transportation)), domain=NonNegativeReals)

# create binary variable ai for whether ODM i is chosen for producing LCD42
model.a = Var(range(len(ODM)), within=(0,1))

# create binary variable bi for whether ODM i is chosen for producing LCD32
model.b = Var(range(len(ODM[0:2])), within=(0,1))


# creates the objective function rule
def objective_rule(model): 
    return sum(d[i]*w[0]*model.x[i,j]*emission[j] for i in range(len(ODM)) 
               for j in range(len(Transportation))) + \
        sum(d[k]*w[1]*model.y[k,j]*emission[j] for k in range(len(ODM[0:2])) 
            for j in range(len(Transportation)))

# adds the objectuve function using the rule
model.obj = Objective(rule=objective_rule, sense=minimize)


# Shipping target constraint
model.target42 = Constraint(expr = sum(model.x[i,j] for i in range(len(ODM)) 
                                       for j in range(len(Transportation))) >= t[0])

model.target32 = Constraint(expr = sum(model.y[k,j] for k in range(len(ODM[0:2])) 
                                       for j in range(len(Transportation))) >= t[1])

# budget constraint
model.budget = Constraint(expr = sum(w[0]*s[i,j]*model.x[i,j] for i in range(len(ODM)) 
           for j in range(len(Transportation))) + \
    sum(w[1]*s[k, j]*model.y[k,j] for k in range(len(ODM[0:2])) 
        for j in range(len(Transportation))) + \
    sum(cx[i]*sum(model.x[i, j] for j in range(len(Transportation))) for i in range(len(ODM))) + \
    sum(cy[k]*sum(model.y[k, j] for j in range(len(Transportation))) for k in range(len(ODM[0:2]))) <= 3000000000)

# Either producing 0 or at least 200000 at most 600000 units if a ODM is chosen

model.upperbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.upperbound.add(sum(model.y[k,j] for j in range(len(Transportation))) <= 600000 * model.b[k])
    
for i in range(2,7):
    model.upperbound.add(sum(model.x[i,j] for j in range(len(Transportation))) <= 600000*model.a[i])

model.lowerbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.a[k])

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.b[k])

for i in range(2,7):
    model.lowerbound.add(sum(model.x[i,j] for j in range(len(Transportation))) >= 200000*model.a[i])
    
    
# Minimum number of products to be shipped via Regular Air or Air Express      
model.air42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(len(Transportation[0:2]))) >= 46000)

model.air32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(len(Transportation[0:2]))) >= 53000)


# Minimum number of products to be shipped via Road or Road LTL or Road Network
model.road42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(2,5)) >= 92000)

model.road32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(2,5)) >= 79500)


# Minimum number of products to be shipped via Rail
model.rail42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(5,6)) >= 138000)

model.rail32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(5,6)) >= 79500)

# Infeasible transportation method
model.infeasible = ConstraintList()
model.infeasible.add(model.x[4,0] == 0)
model.infeasible.add(model.x[4,1] == 0)
model.infeasible.add(model.x[4,5] == 0)
model.infeasible.add(model.x[4,6] == 0)

model.infeasible.add(model.x[5,2] == 0)
model.infeasible.add(model.x[5,3] == 0)
model.infeasible.add(model.x[5,4] == 0)
model.infeasible.add(model.x[5,5] == 0)

solver = SolverFactory('glpk')
results = solver.solve(model,tee=False)

if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Total CO2 emission(kg) is", round(model.obj(),2), "kg")
else:
    print("Solve Failed")
    
for i in range(len(ODM)):
    print("ODM", i+1, "should produce", round(sum(model.x[i,j] for j in range(len(Transportation)))(),0), "units of LCD42"
          )

for i in range(len(ODM)):
    for j in range(len(Transportation)):
            print("ODM", i+1, "should ship", round(model.x[i,j](),0),"LCD42 via", Transportation[j])

for k in range(len(ODM[0:2])):
    print("ODM", k+1, "should produce", sum(model.y[k,j] for j in range(len(Transportation)))(), "units of LCD32"
          )

for k in range(len(ODM[0:2])):
    for j in range(len(Transportation)):
            print("ODM", k+1, "should ship", model.y[k,j](),"LCD32 via", Transportation[j])


for i in range(len(ODM)):
    if model.a[i]() > 0.5:
        print("ODM", i+1, "is selected to produce LCD42")


for k in range(len(ODM[0:2])):
    if model.b[k]() > 0.5:
        print("ODM", k+1, "is selected to produce LCD32")



if results.solver.termination_condition == TerminationCondition.optimal:
    # Calculate the budget used
    budget_used = sum(w[0]*s[i,j]*model.x[i,j].value for i in range(len(ODM)) 
                      for j in range(len(Transportation))) + \
                  sum(w[1]*s[k, j]*model.y[k,j].value for k in range(len(ODM[0:2])) 
                      for j in range(len(Transportation))) + \
                  sum(cx[i]*sum(model.x[i, j].value for j in range(len(Transportation))) 
                      for i in range(len(ODM))) + \
                  sum(cy[k]*sum(model.y[k, j].value for j in range(len(Transportation))) 
                      for k in range(len(ODM[0:2])))

    print("Budget used:", round(budget_used,2))



"""
Minimizing CO2 emission under 3.3 billion budget

Data:
    di: the distance between ODM i and the distribution centre
    cxi: the production cost of LCD42 at ODM i
    cyi: the production cost of LCD32 at ODM i
    w: product weight in metric ton
    t: units of each product to be shipped
    sij: shipping cost from ODM i via transportation j
    emission,j: CO2 emission in kg for transportation j per ton-km shipped
    
Decision Variable:
    xij: the amount of LCD42 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    yij: the amount of LCD32 produced by ODM i shipped via transportation j
            for i in 1-7; for j in 1-7.
    ai: whether ODM i is selected to produce LCD42
    bi: whether ODM i is selected to produce LCD32
    

"""
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import pandas as pd
import numpy as np

d = [2508, 1553, 1380, 2150, 30, 690, 686]
cx = [1983.4, 2254, 2582.4, 1976.1, 2711.3, 2704.8, 2125.2]
cy = [1818, 1996.4]
w = [0.022, 0.0165]
t = [920000, 530000]
s = [64400, 70840, 6182.4, 5216.4, 4830, 4250.4, 3091.2,
     115920, 127512, 7084, 5796, 5667.2, 5796, 2704.8,
     103040, 113344, 7084, 5796, 5667.2, 5796, 3284.4,
     64400, 70840, 6182.4, 5280.8, 5216.4, 4250.4, 3091.2,
     0, 0, 9660, 9016, 8694, 0, 0,
     135240, 148120, 0, 0, 0, 0, 3413.2,
     103040, 112700, 7084, 5796, 5538.4, 5860.4, 2729.2]
emission = [1.44, 1.44, 0.0613, 0.0613, 0.0613, 0.0285, 0.007]


ODM = ("ODM1", "ODM2", "ODM3", "ODM4", "ODM5", "ODM6", "ODM7")
Transportation = ("Regular Air", "Air Express", "Road", "Road LTL", "Road Network", "Rail", "Water")


# turning shipping cost list into a 7*7 table
s = np.array(s).reshape(7, 7)

# create a blanck concrete model
model = ConcreteModel()

# create variable xij for the amount of LCD42 produced by ODM i shipped via transportation j
model.x = Var(range(len(ODM)), range(len(Transportation)), domain=NonNegativeReals)
 
# creates variable yij for the amount of LCD32 produced by ODM i shipped via transportation j
model.y = Var(range(len(ODM[0:2])), range(len(Transportation)), domain=NonNegativeReals)

# create binary variable ai for whether ODM i is chosen for producing LCD42
model.a = Var(range(len(ODM)), within=(0,1))

# create binary variable bi for whether ODM i is chosen for producing LCD32
model.b = Var(range(len(ODM[0:2])), within=(0,1))


# creates the objective function rule
def objective_rule(model): 
    return sum(d[i]*w[0]*model.x[i,j]*emission[j] for i in range(len(ODM)) 
               for j in range(len(Transportation))) + \
        sum(d[k]*w[1]*model.y[k,j]*emission[j] for k in range(len(ODM[0:2])) 
            for j in range(len(Transportation)))

# adds the objectuve function using the rule
model.obj = Objective(rule=objective_rule, sense=minimize)


# Shipping target constraint
model.target42 = Constraint(expr = sum(model.x[i,j] for i in range(len(ODM)) 
                                       for j in range(len(Transportation))) >= t[0])

model.target32 = Constraint(expr = sum(model.y[k,j] for k in range(len(ODM[0:2])) 
                                       for j in range(len(Transportation))) >= t[1])

# budget constraint
model.budget = Constraint(expr = sum(w[0]*s[i,j]*model.x[i,j] for i in range(len(ODM)) 
           for j in range(len(Transportation))) + \
    sum(w[1]*s[k, j]*model.y[k,j] for k in range(len(ODM[0:2])) 
        for j in range(len(Transportation))) + \
    sum(cx[i]*sum(model.x[i, j] for j in range(len(Transportation))) for i in range(len(ODM))) + \
    sum(cy[k]*sum(model.y[k, j] for j in range(len(Transportation))) for k in range(len(ODM[0:2]))) <= 3300000000)

# Either producing 0 or at least 200000 at most 600000 units if a ODM is chosen

model.upperbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.upperbound.add(sum(model.y[k,j] for j in range(len(Transportation))) <= 600000 * model.b[k])
    
for i in range(2,7):
    model.upperbound.add(sum(model.x[i,j] for j in range(len(Transportation))) <= 600000*model.a[i])

model.lowerbound = ConstraintList()

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.a[k])

for k in range(len(ODM[0:2])):
    model.lowerbound.add(sum(model.x[k,j]+model.y[k,j] for j in range(len(Transportation))) >= 200000*model.b[k])

for i in range(2,7):
    model.lowerbound.add(sum(model.x[i,j] for j in range(len(Transportation))) >= 200000*model.a[i])
    
    
# Minimum number of products to be shipped via Regular Air or Air Express      
model.air42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(len(Transportation[0:2]))) >= 46000)

model.air32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(len(Transportation[0:2]))) >= 53000)


# Minimum number of products to be shipped via Road or Road LTL or Road Network
model.road42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(2,5)) >= 92000)

model.road32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(2,5)) >= 79500)


# Minimum number of products to be shipped via Rail
model.rail42 = Constraint(expr = sum(model.x[i, j] for i in range(len(ODM)) for j in range(5,6)) >= 138000)

model.rail32 = Constraint(expr = sum(model.y[k, j] for k in range(len(ODM[0:2])) for j in range(5,6)) >= 79500)

# Infeasible transportation method
model.infeasible = ConstraintList()
model.infeasible.add(model.x[4,0] == 0)
model.infeasible.add(model.x[4,1] == 0)
model.infeasible.add(model.x[4,5] == 0)
model.infeasible.add(model.x[4,6] == 0)

model.infeasible.add(model.x[5,2] == 0)
model.infeasible.add(model.x[5,3] == 0)
model.infeasible.add(model.x[5,4] == 0)
model.infeasible.add(model.x[5,5] == 0)

solver = SolverFactory('glpk')
results = solver.solve(model,tee=False)

if(results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Total CO2 emission(kg) is", round(model.obj(),2), "kg")
else:
    print("Solve Failed")
    
for i in range(len(ODM)):
    print("ODM", i+1, "should produce", round(sum(model.x[i,j] for j in range(len(Transportation)))(),0), "units of LCD42"
          )

for i in range(len(ODM)):
    for j in range(len(Transportation)):
            print("ODM", i+1, "should ship", round(model.x[i,j](),0),"LCD42 via", Transportation[j])

for k in range(len(ODM[0:2])):
    print("ODM", k+1, "should produce", sum(model.y[k,j] for j in range(len(Transportation)))(), "units of LCD32"
          )

for k in range(len(ODM[0:2])):
    for j in range(len(Transportation)):
            print("ODM", k+1, "should ship", model.y[k,j](),"LCD32 via", Transportation[j])


for i in range(len(ODM)):
    if model.a[i]() > 0.5:
        print("ODM", i+1, "is selected to produce LCD42")


for k in range(len(ODM[0:2])):
    if model.b[k]() > 0.5:
        print("ODM", k+1, "is selected to produce LCD32")



if results.solver.termination_condition == TerminationCondition.optimal:
    # Calculate the budget used
    budget_used = sum(w[0]*s[i,j]*model.x[i,j].value for i in range(len(ODM)) 
                      for j in range(len(Transportation))) + \
                  sum(w[1]*s[k, j]*model.y[k,j].value for k in range(len(ODM[0:2])) 
                      for j in range(len(Transportation))) + \
                  sum(cx[i]*sum(model.x[i, j].value for j in range(len(Transportation))) 
                      for i in range(len(ODM))) + \
                  sum(cy[k]*sum(model.y[k, j].value for j in range(len(Transportation))) 
                      for k in range(len(ODM[0:2])))

    print("Budget used:", round(budget_used,2))


