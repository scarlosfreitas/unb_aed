import datetime
import pandas as pd
import pulp

# Problem: Unrelated-Machines Scheduling with Integer Linear Programming (ILP)

# Dataframes preparation
timer = datetime.datetime.now()

csv_tasks = 'datasets/tasks.csv'
csv_consultants = 'datasets/consultants.csv'
csv_productivity = 'datasets/productivity.csv'

n_task_limit = 15
n_tasks_fixed = n_task_limit - 9
n_jobs = 14
offset = 380/3

# print('Memoria:',memory_usage())

# Prepair list os time os each task
df_tasks = pd.read_csv(csv_tasks)
list_tasks = df_tasks['time'].to_list()[0:n_task_limit] # Estou limintando as tarefas para acelerar
matrix_tasks = np.diag(np.array(list_tasks))

# Prepair list of consultants cost per hour
df_consultants_cost = pd.read_csv(csv_consultants)
list_consultants_cost = df_consultants_cost['cost'].to_list()
# matrix_consultants_cost = np.diag(np.array(list_consultants_cost))

# Prepair the matrix of produtivity
df_productivity = pd.read_csv(csv_productivity)
matrix_productivity = df_productivity.iloc[:n_task_limit,1:].to_numpy()
matrix_task_duration = np.dot(matrix_tasks,matrix_productivity)

print('Datasets prepairation complete :',datetime.datetime.now() - timer)

# print('Memoria:',memory_usage())

# Combinations
timer = datetime.datetime.now()

m_consultants = len(list_consultants_cost)
n_tasks = len(list_tasks)
k_combinations = m_consultants ** n_tasks


P = matrix_task_duration.transpose()

num_machines, num_jobs = P.shape

# Step 1: Define the ILP problem
problem = pulp.LpProblem("Unrelated_Machines_Scheduling", pulp.LpMinimize)

# Step 2: Define decision variables
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_machines) for j in range(num_jobs)), cat='Binary')

# Makespan variable (objective to minimize)
T = pulp.LpVariable("Makespan", lowBound=0)

# Step 3: Objective function: minimize makespan
problem += T

# Step 4: Constraints

# Each job is assigned to exactly one machine
for j in range(num_jobs):
    problem += pulp.lpSum(x[i, j] for i in range(num_machines)) == 1

# The total processing time on each machine should not exceed the makespan
for i in range(num_machines):
    problem += pulp.lpSum(P[i][j] * x[i, j] for j in range(num_jobs)) <= T

# Step 5: Solve the problem
problem.solve()

# Step 6: Output results
print("Status:", pulp.LpStatus[problem.status])
print("Optimal Makespan:", pulp.value(T))

print("\nJob Assignments (job: machine):")
for j in range(num_jobs):
    for i in range(num_machines):
        if pulp.value(x[i, j]) == 1:
            print(f"Job {j}: Machine {i}")

print('Execução total: ',datetime.datetime.now()-timer)