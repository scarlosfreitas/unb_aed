import heapq
import numpy as np
import datetime
import pandas as pd
import itertools
import json
from joblib import Parallel, delayed
from memory_profiler import profile
from memory_profiler import memory_usage
import argparse

# Criando um parser
parser = argparse.ArgumentParser(description="Processa tarefas com multiprocessamento e limitação de tarefas")
# Adicionando argumentos
parser.add_argument('--n_task_limit', type=int, help='Quantidade limite de tarefas')
parser.add_argument('--n_tasks_fixed', type=int, help='Quantidade que é fixada e distribuida entre jobs')
parser.add_argument('--n_jobs', type=int, help='Quantidade que é fixada e distribuida entre jobs')

# Lendo os argumentos
args = parser.parse_args()

# Functions
# Create a iterator with the combinations of m consultants and n tasks
def get_iter_combinations(m_consultants,n_tasks):
    return list(itertools.product(range(0,m_consultants), repeat=n_tasks))

# Process the list of combnations
def brute_force_multiprocess(combination_fixed,m_consultants,n_tasks,matrix_task_duration,offset=None):
    # print('task_aloc_0,task_aloc_1',task_aloc_0,task_aloc_1)
    # matrix_task_duration, 

    iter_combinations = get_iter_combinations(m_consultants,n_tasks-len(combination_fixed))

    # print(matriz_custo_esperado)
    project_duration_minimal = None
    project_alocation_minimal = []
    project_consulter_hours_minimal = []

    for combination in iter_combinations:
        project_duration = 0
        project_alocation = list(combination_fixed) + list(combination)
        project_consulter_hours = [0]*m_consultants

        for n in range(0,n_tasks):
            project_consulter_hours[project_alocation[n]] += matrix_task_duration[n][project_alocation[n]]
        
        project_duration = max(project_consulter_hours)

        if offset is None or project_duration <= offset:
            if project_duration_minimal is None or project_duration < project_duration_minimal:
                project_duration_minimal = project_duration
                project_alocation_minimal = [project_alocation]
                project_consulter_hours_minimal = [project_consulter_hours]
            elif project_duration == project_duration_minimal:
                project_alocation_minimal.append(project_alocation)
                project_consulter_hours_minimal.append(project_consulter_hours)

    
    print('Job finished {} | Memory: {:.1f} Mb | project_duration: {} | soluctions: {}'.format(
        combination_fixed,
        memory_usage()[0],
        project_duration_minimal,
        len(project_alocation_minimal)))
    return {
        'duration': project_duration_minimal, 
        'alocations': project_alocation_minimal, 
        'hour_per_consultant': project_consulter_hours_minimal
        }







# Dataframes preparation
timer = datetime.datetime.now()

csv_tasks = 'datasets/tasks.csv'
csv_consultants = 'datasets/consultants.csv'
csv_productivity = 'datasets/productivity.csv'

n_task_limit = args.n_task_limit or 12 # 15
n_tasks_fixed = args.n_tasks_fixed or n_task_limit - 9 # 9
if n_tasks_fixed < 0:
    n_tasks_fixed = 0
n_jobs = args.n_jobs or 8 # 14
offset = 380/3

print('n_task_limit: {}, n_tasks_fixed: {}, n_jobs: {}, offset: {}'.format(n_task_limit,n_tasks_fixed,n_jobs,offset))

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

# Combinations
timer = datetime.datetime.now()

m_consultants = len(list_consultants_cost)
n_tasks = len(list_tasks)
# n_tasks_fixed = 2
k_combinations = m_consultants ** n_tasks

iter_combinations_fixed = get_iter_combinations(m_consultants,n_tasks_fixed)

print('Number of combinations: {:,.0f}'.format(k_combinations))
print('Number of tasks: {} fixed: {} worker: {}'.format(n_tasks,n_tasks_fixed,n_tasks-n_tasks_fixed))

print('Memoria:',memory_usage())

timer = datetime.datetime.now()

# Parallel Process
list_results = Parallel(n_jobs=n_jobs)(delayed(brute_force_multiprocess)(
    combination_fixed,
    m_consultants,
    n_tasks,matrix_task_duration,offset) for combination_fixed in iter_combinations_fixed)

print('Paralelel processa finished')

# Take the best alocations
best_duration = None
best_results = []

for result in list_results:
    if result['duration'] is not None:
        if best_duration is None or result['duration'] < best_duration:
            best_duration = result['duration']
            best_results = [result]
        elif result['duration'] == best_duration:
            best_results.append(result)

total_duration = datetime.datetime.now() - timer
process_capacity = round(k_combinations/total_duration.total_seconds())

json_result = {
    'duration': best_duration,
    'cost': best_duration * sum(list_consultants_cost),
    'combinations':k_combinations,
    'n_task':n_task_limit,
    'n_tasks_fixed':n_tasks_fixed,
    'n_jobs':n_jobs,
    'memory':memory_usage()[0],
    'total_duration':total_duration.total_seconds(),
    'combination_per_second':process_capacity,
    'alocations':[alocation for result in best_results for alocation in result['alocations']],
    'hour_per_consultant':[hour_per_consultant for result in best_results for hour_per_consultant in result['hour_per_consultant']],
}

print('Brute Force Process: {:,.0f} combinations/second | Total Duration: {}'.format(process_capacity,total_duration))

with open('brute_force_result_t{}_j{}_f{}.json'.format(n_task_limit,n_jobs,n_tasks_fixed), 'w') as arquivo:
    json.dump(json_result, arquivo, 
              indent=4, 
              ensure_ascii=False)

print('duration:',json_result['duration'])
print('cost:',json_result['cost'])
print('alocations:',json_result['alocations'])
