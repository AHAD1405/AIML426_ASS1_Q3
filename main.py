import operator
import math
import random
import numpy as np
import pandas as pd

# Define the primitive set
# Define the function set and terminal set
FUNCTIONS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': lambda x, y: x / y if y != 0 else 1, # Avoid division by zero
    'sin': math.sin
}
TERMINALS = ['x', 1, 3.0, 2.0]


# Generate a random expression
def generate_random_expression(depth, seed_=42 ):
    #random.seed(seed_)
    if depth == 0:
        return random.choice(TERMINALS)
    else:
        func = random.choice(list(FUNCTIONS.keys()))
        if func in ['add', 'sub', 'mul', 'div']:
            return [func, generate_random_expression(depth - 1), generate_random_expression(depth - 1)]
        else:
            return [func, generate_random_expression(depth - 1)]

# Crossover
def crossover(parent1, parent2):
    if isinstance(parent1, list) and isinstance(parent2, list):
        if parent1[0] in ['sin'] or parent2[0] in ['sin']:  # Ensure unary functions don't receive two arguments
            return parent1 if random.random() < 0.5 else parent2
        if random.random() < 0.5:
            return [parent1[0], crossover(parent1[1], parent2[1]), crossover(parent1[2], parent2[2])]
        else:
            return [parent2[0], crossover(parent1[1], parent2[1]), crossover(parent1[2], parent2[2])]
    else:
        return parent1 if random.random() < 0.5 else parent2


# Selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda ind_fit: ind_fit[1])
    return selected[0][0]

# Define the target function
def target_function(x):
    if x > 0:
        return 1 / x + math.sin(x)
    else:
        return 2 * x + x**2 + 3.0
    
# Evaluate an expression
def evaluate_expression(expr, x):
    if isinstance(expr, list):
        func = FUNCTIONS[expr[0]]
        if len(expr) == 2:  # Unary function
            return func(evaluate_expression(expr[1], x))
        elif len(expr) == 3:  # Binary functions
            return func(evaluate_expression(expr[1], x), evaluate_expression(expr[2], x))
    elif expr == 'x':
        return x
    else:
        return expr

def fitness(individual, points):
    sqerrors = [(evaluate_expression(individual, x) - target_function(x)) **2 for x in points]
    return sum(sqerrors) / len(points)

# Mutation
def mutate(individual, depth):
    if random.random() < 0.1:
        return generate_random_expression(depth)
    elif isinstance(individual, list):
        if len(individual) == 2:  # Unary function
            return [individual[0], mutate(individual[1], depth - 1)]
        elif len(individual) == 3:  # Binary function
            return [individual[0], mutate(individual[1], depth - 1), mutate(individual[2], depth - 1)]
    return individual

def count_nodes(expr):
    if isinstance(expr, list):
        # Count the current node and recursively count the nodes in the arguments
        return 1 + sum(count_nodes(arg) for arg in expr[1:])
    else:
        # Leaf node (variable or constant)
        return 1

def print_table(best_fitness, best_individual):
    # calculate mean and std
    best_fitness_mean = np.mean(best_fitness)
    best_fitness_std = np.std(best_fitness)

    # calculate nodes 
    nodes = [round(count_nodes(ind)) for ind in best_individual]
    # calculate depth
    #depth = [calculate_depth(ind) for ind in best_individual]

    # Calculate mean and Std for nodes and depth
    nodes_mean = np.mean(nodes)
    nodes_std = np.std(nodes)
    # First column
    first_column = ['Run 1','Run 2','Run 3']

    # Create a dictionary with the two lists as values
    data = {'': first_column, 'Best Fitness': best_fitness, 'Program Size': nodes}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'Best Fitness': [best_fitness_mean], 'Program Size': [nodes_mean]})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Best Fitness': [best_fitness_std], 'Program Size': [nodes_std]})
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table



def main():
    population_size = 100
    generations = 40
    tournament_size = 5
    crossover_rate = 0.9
    seeds_val = [20, 30,40]
    max_depth = 5
    best_fitness_li = []
    best_individual_li = []
    points = np.linspace(-5, 5, 100)
    run_no = 3

    for run in range(run_no):
        print(f"Run {run + 1} of {run_no} . . . ")

        # Initialize the population
        population = [generate_random_expression(max_depth) for _ in range(population_size)]
        best_individual = None
        best_fitness = float('inf')

        for generation in range(generations):
            print(f"\tGeneration {generation + 1} . . . ")

            fitnessses = [round(fitness(individual, points), 4) for individual in population]
            
            offspring_population = []
            while len(offspring_population) < population_size:
                parent1 = tournament_selection(population, fitnessses, tournament_size)
                parent2 = tournament_selection(population, fitnessses, tournament_size)
                if random.random() < crossover_rate:
                    child = crossover(parent1, parent2)
                else:
                    child = parent1
                child = mutate(child, max_depth)
                offspring_population.append(child)
            
            population = offspring_population

            # Track the best individual of the current generation
            current_best_individual = min(population, key=lambda ind: fitness(ind, points))
            current_best_fitness = round(fitness(current_best_individual, points), 4)
            if current_best_fitness < best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
            
        # Print the best individual of the current generation
        #best_individual = min(population, key=lambda ind: fitness(ind, points))
        #best_fitness = fitness(best_individual, points)
        best_individual_li.append(best_individual)
        best_fitness_li.append(round(best_fitness, 4))
        print('-------------------------------------------------')
    
    # Print Summary after claculate means and std
    data_table = print_table(best_fitness_li, best_individual_li)
    print(data_table)


if __name__ == "__main__":
    main()
