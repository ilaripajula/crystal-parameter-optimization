# External libraries
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
# Our classes
from classes.SIM import *
from classes.preprocessing import *

# -------------------------------------------------------------------
#   Define parameter range.
# -------------------------------------------------------------------

param_range = {
    'tau':{'low': 660, 'high': 700, 'step': 1},
    'taucs':{'low': 500, 'high': 1000, 'step': 50}, 
    'h0':{'low': 30000, 'high': 50000, 'step': 500},
    'alpha':{'low': 3, 'high': 5, 'step': 0.5}
}

# -------------------------------------------------------------------
#   Run initial simulations (used to initialize optimization process).
# -------------------------------------------------------------------

# Initialize SIM object.
sim = SIM(param_range)
print("Running Initial Simulations...")
sim.run_initial_simulations()
print(f"Done. {len(sim.simulations)} simulations completed.")
np.save('initial_simulations.npy', sim.simulations)

# -------------------------------------
#   Set up experimental curve (target)
# -------------------------------------

exp = pd.read_csv('CP2_Validation.csv')
exp_stress = exp.iloc[:,0]
exp_strain = exp.iloc[:,1]
#sim_strain = exp['Simulated strain'].dropna()
f = interp1d(exp_strain, exp_stress)
x_min, x_max = 0.002, exp_strain.max()
prune = np.logical_and(sim.strain > x_min, sim.strain < x_max)
sim.strain = sim.strain[prune]
exp_target = f(sim.strain).reshape(1,len(sim.strain))

# -----------------------------------------
#  Initialize Response Surface Module (MLP)
# -----------------------------------------

mlp = MLPRegressor(hidden_layer_sizes=[15], solver='adam', max_iter=100000, shuffle=True)
print("Fitting response surface...")
y = np.array([stress[prune] for (_, stress) in sim.simulations.values()])
X = np.array(list(sim.simulations.keys()))
mlp.fit(X,y)

# -------------------------------
#      Initialize GA.
# -------------------------------

# Initialize fitness function
def fitness(solution, solution_idx):
    sol = solution.reshape((1,4))
    y_pred = mlp.predict(sol)
    fitness = 1 / mean_squared_error(y_pred, exp_target)
    return fitness

# Initialize GA Optimizer
num_generations = 100 # Number of generations.
num_parents_mating = 500 # Number of solutions to be selected as parents in the mating pool.
sol_per_pop = 1000 # Number of solutions in the population.
num_genes = 4
gene_space = [param_range['tau'], param_range['taucs'], param_range['h0'], param_range['alpha']]
last_fitness = 0
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 25

def on_generation(ga_instance):
    global last_fitness
    generation = ga_instance.generations_completed
    fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness,
                       on_generation=on_generation,
                       gene_space=gene_space,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)


# -------------------------------
#      Helper Functions
# -------------------------------

def output_results(ga_instance):
    # Returning the details of the best solution in a dictionary.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    best_solution_generation = ga_instance.best_solution_generation
    loss = 1/solution_fitness
    values = (solution, solution_fitness, solution_idx, best_solution_generation, loss)
    keys = ("solution", "solution_fitness", "solution_idx", "best_solution_generation", "loss")
    output = dict(zip(keys, values))
    return output

def print_results(results):
    print(f"Parameters of the best solution : {results['solution']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Index of the best solution : {results['solution_idx']}")
    print(f"MSE given by the MLP estimate: {results['loss']}")
    print(f"Best fitness value reached after {results['best_solution_generation']} generations.")

# -------------------------------
#      Optimization Loop
# -------------------------------
epsilon = 10.0 # Target MSE value.

# Optimization on initial simulations.
print("Optimizing using GA...")
ga_instance.run()
results = output_results(ga_instance)
print_results(results)
loss = 1e6
# Iterative optimization.
print("--------------------------------")
print("Starting iterative optimization:")
while loss > epsilon:
    sim.run_single_test(tuple(results['solution']))
    np.save('simulations.npy', sim.simulations)
    y = np.array([stress[prune] for (_, stress) in sim.simulations.values()])
    X = np.array(list(sim.simulations.keys()))
    mlp.fit(X,y)
    ga_instance.run()
    results = output_results(ga_instance)
    print_results(results)
    loss = mean_squared_error(y[-1].reshape((1,-1)), exp_target)
    print(f"LOSS = {loss}")

print("Optimization Complete")
print("--------------------------------")
print("Final Parameters: ", results['solution'])
print(f"MSE of the final solution : {loss}")