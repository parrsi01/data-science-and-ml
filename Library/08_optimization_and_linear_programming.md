# Optimization and Linear Programming

---

> **Field** — Operations Research, Mathematical Programming
> **Scope** — Linear optimization, resource allocation,
> constraint modeling, and solver tools for data science

---

## Overview

Optimization is the process of finding the best possible
solution from a set of available options, subject to
rules you define. Linear programming is the most common
form: you write a goal (like "minimize cost") and a set
of constraints (like "don't exceed the budget"), then a
solver finds the answer. This is used everywhere from
scheduling shifts to allocating server resources.

---

## Definitions

### `Linear Programming`

**Definition.**
Linear programming (LP) is a method for finding the
best outcome in a mathematical model where the goal
and all constraints are expressed as straight-line
(linear) equations. "Best" means either the maximum
or minimum value of something you care about.

**Context.**
LP is one of the most practical tools in data science.
Whenever you need to allocate limited resources
(money, time, people, machines) in the most efficient
way, LP gives you a proven, repeatable method.

**Example.**
A company wants to minimize shipping costs across
three warehouses while meeting demand at five stores.
LP finds the cheapest allocation.

```python
from pulp import LpProblem, LpMinimize

model = LpProblem("shipping", LpMinimize)
```

---

### `Objective Function`

**Definition.**
The objective function is the formula that describes
what you want to optimize. It is the single equation
you are trying to maximize or minimize. Every LP
problem has exactly one objective function.

**Context.**
This is the heart of any optimization problem. Before
writing code, you must decide: what am I trying to
make as large or as small as possible? Common
objectives include minimizing cost, maximizing profit,
or minimizing wasted time.

**Example.**
If you are assigning workers to shifts and want to
minimize total labor cost:

```
Objective: Minimize (wage_A * hours_A
                   + wage_B * hours_B
                   + wage_C * hours_C)
```

In PuLP:

```python
model += (
    wage_a * hours_a
    + wage_b * hours_b
    + wage_c * hours_c,
    "Total_Labor_Cost"
)
```

---

### `Constraint`

**Definition.**
A constraint is a rule that limits what solutions
are allowed. It is written as a linear inequality
or equality. Constraints define the boundaries of
what is possible in your problem.

**Context.**
Without constraints, optimization is trivial (just
set everything to zero or infinity). Real problems
have limits: budgets, capacities, minimum service
levels, legal requirements. Each of these becomes
a constraint in your model.

**Example.**
"Each warehouse can ship at most 500 units per day"
becomes:

```python
model += (
    shipments_warehouse_A <= 500,
    "Warehouse_A_Capacity"
)
```

You can have hundreds of constraints in a single
model. The solver handles them all simultaneously.

---

### `Feasible Region`

**Definition.**
The feasible region is the set of all solutions that
satisfy every constraint at the same time. It is the
"allowed zone" in which the solver searches for the
best answer. If you plotted it on a graph, it would
be the area where all constraint lines overlap.

**Context.**
Understanding feasibility is essential for debugging.
If your model returns no solution, it usually means
the feasible region is empty (your constraints
conflict with each other). Visualizing the feasible
region helps you understand what trade-offs exist.

**Example.**
Imagine two constraints:

- x + y <= 10
- x >= 2
- y >= 3

The feasible region is every (x, y) point that
satisfies all three rules at once. The optimal
solution sits at one corner of this region.

```
y
|
|   *-------*
|   |       |
|   |  OK   |
|   *-------*
+---+-------+--- x
    2       7
```

---

### `Decision Variable`

**Definition.**
A decision variable is something you can control
and want the solver to figure out for you. These
are the unknowns in your model. The solver assigns
values to each decision variable to produce the
best solution.

**Context.**
Identifying your decision variables is the first
step in formulating an LP problem. Ask yourself:
"What choices am I making?" Each choice becomes
a decision variable. The solver then decides the
value of each one.

**Example.**
If you are deciding how many units to produce at
each factory:

```python
from pulp import LpVariable

units_factory_A = LpVariable(
    "units_A", lowBound=0, cat="Integer"
)
units_factory_B = LpVariable(
    "units_B", lowBound=0, cat="Integer"
)
```

Here, `units_factory_A` and `units_factory_B`
are the decision variables. The solver finds
their optimal values.

---

### `Sensitivity Analysis`

**Definition.**
Sensitivity analysis tells you how much your
optimal solution would change if you adjusted
one of the inputs slightly. It answers the
question: "How fragile is this answer?" It does
not require re-solving the entire problem.

**Context.**
In real projects, your input data is never
perfectly accurate. Sensitivity analysis shows
which parameters matter most. If changing a cost
by 5% flips the entire solution, you know you
need to measure that cost very carefully.

**Example.**
After solving a model in PuLP, you can inspect
the shadow prices (dual values) of constraints:

```python
for name, c in model.constraints.items():
    print(name, c.pi)
```

A high shadow price on a constraint means that
relaxing that constraint slightly would improve
your objective significantly.

---

### `Fairness Metric`

**Definition.**
A fairness metric is a numerical measure of how
evenly resources or outcomes are distributed
across groups. In optimization, fairness
constraints prevent the solver from favoring one
group at the expense of another.

**Context.**
Pure optimization can produce solutions that are
technically "optimal" but socially unfair. For
example, allocating all resources to the most
profitable region while starving others. Fairness
metrics let you balance efficiency with equity.

**Example.**
One common fairness metric is the ratio between
the best-served and worst-served group:

```python
# Ensure no region gets less than 80%
# of the average allocation
for region in regions:
    model += (
        allocation[region]
        >= 0.8 * total / len(regions),
        f"Fairness_{region}"
    )
```

Other approaches include the Gini coefficient,
max-min fairness, or proportional fairness.

---

### `PuLP`

**Definition.**
PuLP is an open-source Python library for defining
and solving linear programming problems. It lets
you write optimization models in plain Python
syntax and automatically connects to a solver
engine to find the answer.

**Context.**
PuLP is the most beginner-friendly LP library in
Python. It is widely used in data science courses
and industry projects. You write your model using
Python variables and operators, and PuLP handles
all the math behind the scenes.

**Example.**
A complete minimal PuLP model:

```python
from pulp import (
    LpProblem, LpMinimize,
    LpVariable, LpStatus, value
)

# Create the model
model = LpProblem("example", LpMinimize)

# Decision variables
x = LpVariable("x", lowBound=0)
y = LpVariable("y", lowBound=0)

# Objective
model += 2 * x + 3 * y, "Cost"

# Constraints
model += x + y >= 10, "Minimum_Total"
model += x <= 8, "Max_X"

# Solve
model.solve()
print(LpStatus[model.status])
print(f"x = {value(x)}, y = {value(y)}")
```

Install with:

```bash
pip install pulp
```

---

### `CBC Solver`

**Definition.**
CBC (Coin-or Branch and Cut) is the default
open-source solver that ships with PuLP. It is a
free, general-purpose solver capable of handling
linear programming and mixed-integer programming
problems.

**Context.**
When you call `model.solve()` in PuLP without
specifying a solver, CBC runs automatically. For
most academic and medium-scale problems, CBC is
fast enough. For very large industrial problems,
commercial solvers like Gurobi or CPLEX may be
needed.

**Example.**
Using CBC explicitly in PuLP:

```python
from pulp import PULP_CBC_CMD

model.solve(PULP_CBC_CMD(msg=True))
```

The `msg=True` flag prints the solver's progress
log, which is useful for debugging slow models.

You can also set a time limit:

```python
model.solve(PULP_CBC_CMD(timeLimit=60))
```

This stops the solver after 60 seconds and returns
the best solution found so far.

---

### `Allocation Problem`

**Definition.**
An allocation problem is any optimization task where
you must distribute limited resources across multiple
recipients or uses. It asks: "Given what I have, what
is the best way to divide it up?"

**Context.**
Allocation problems are everywhere in data science
and operations. Examples include assigning budgets
to marketing channels, distributing vaccines to
regions, scheduling staff across shifts, or routing
packages through a delivery network.

**Example.**
Distributing 1000 units of inventory across 5 stores
to maximize total expected sales:

```python
from pulp import LpProblem, LpMaximize, LpVariable

model = LpProblem("inventory", LpMaximize)

stores = ["A", "B", "C", "D", "E"]
alloc = {
    s: LpVariable(f"alloc_{s}", lowBound=0)
    for s in stores
}

# Total inventory constraint
model += (
    sum(alloc[s] for s in stores) <= 1000,
    "Total_Inventory"
)

# Objective: maximize expected sales
# (each store has a different sales rate)
rates = {"A": 1.2, "B": 0.9, "C": 1.5,
         "D": 0.7, "E": 1.1}
model += sum(
    rates[s] * alloc[s] for s in stores
)
```

---

### `Infeasible Solution`

**Definition.**
An infeasible solution means there is no possible
assignment of decision variables that satisfies
all constraints simultaneously. The model has no
valid answer because the rules contradict each
other.

**Context.**
Infeasibility is one of the most common errors when
building optimization models. It usually means you
have been too restrictive with your constraints.
Debugging infeasibility is a critical skill: you
need to find which constraints conflict and relax
or remove one of them.

**Example.**
This model is infeasible because the constraints
are contradictory:

```python
from pulp import LpProblem, LpMinimize, LpVariable

model = LpProblem("broken", LpMinimize)
x = LpVariable("x", lowBound=0)

model += x, "Objective"

# Contradictory constraints
model += x >= 10, "Must_Be_At_Least_10"
model += x <= 5, "Must_Be_At_Most_5"

model.solve()
# Status: Infeasible
```

To debug, try removing constraints one at a time
until the model becomes feasible. The last removed
constraint is part of the conflict.

---

### `Optimal Solution`

**Definition.**
The optimal solution is the set of decision variable
values that produces the best possible objective
function value while satisfying all constraints.
"Best" means lowest (for minimization) or highest
(for maximization).

**Context.**
Finding the optimal solution is the entire point of
optimization. In linear programming, the optimal
solution (if it exists) always occurs at a vertex
(corner point) of the feasible region. The solver
efficiently searches these corner points.

**Example.**
After solving a PuLP model, you extract the optimal
solution like this:

```python
from pulp import value, LpStatus

model.solve()

# Check if optimal
print(f"Status: {LpStatus[model.status]}")

# Print optimal objective value
print(f"Best cost: {value(model.objective)}")

# Print optimal variable values
for v in model.variables():
    print(f"  {v.name} = {v.varValue}")
```

The status will be "Optimal" if a solution was found.
Other statuses include "Infeasible", "Unbounded",
and "Not Solved".

---

### `Relaxation`

**Definition.**
Relaxation means loosening one or more constraints
to make a problem easier to solve. The most common
form is LP relaxation, where integer requirements
are removed so variables can take fractional values.
This gives a bound on how good the true answer
could be.

**Context.**
Relaxation is used in two main situations. First,
when debugging an infeasible model, you relax
constraints to find which ones are causing the
conflict. Second, in mixed-integer programming,
solving the LP relaxation gives you a lower bound
(for minimization) that helps gauge solution quality.

**Example.**
Original integer constraint:

```python
x = LpVariable("x", lowBound=0, cat="Integer")
```

Relaxed (continuous) version:

```python
x = LpVariable("x", lowBound=0, cat="Continuous")
```

The relaxed solution might say x = 3.7, while the
true integer solution is x = 4. The relaxed
objective value tells you how much optimality you
lose by requiring whole numbers.

You can also relax specific constraints by
increasing their right-hand side:

```python
# Original: capacity <= 100
# Relaxed:  capacity <= 120
model += capacity <= 120, "Relaxed_Capacity"
```

---

## See Also

- [Statistical Foundations](./01_statistical_foundations.md)
- [Scaling and Distributed Processing](./11_scaling_and_distributed_processing.md)
- [Reproducibility and Governance](./14_reproducibility_and_governance.md)

---

> **Author** — Simon Parris | Data Science Reference Library
