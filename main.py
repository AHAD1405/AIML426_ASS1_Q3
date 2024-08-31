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
    'sin': math.sin,
    'exp': math.exp
}
TERMINALS = ['x', 1, 3.0, 2.0]


# Generate a random expression
def generate_random_expression(depth):
    if depth == 0:
        return random.choice(TERMINALS)
    else:
        func = random.choice(list(FUNCTIONS.keys()))
        if func in ['sin', 'exp']:
            return [func, generate_random_expression(depth - 1)]
        else:
            return [func, generate_random_expression(depth - 1), generate_random_expression(depth - 1)]

# Crossover
def crossover(parent1, parent2):
    if isinstance(parent1, list) and isinstance(parent2, list):
        if random.random() < 0.5:
            return [parent1[0], crossover(parent1[0], parent2[0]), crossover(parent1[1], parent2[1])]
        else:
            return [parent2[0], crossover(parent1[0], parent2[0]), crossover(parent1[1], parent2[1])]
    else:
        return parent1 if random.random() < 0.5 else parent2

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
        if len(expr) == 2:
            return func(evaluate_expression(expr[1], x))
        elif len(expr) == 3:
            return func(evaluate_expression(expr[1], x), evaluate_expression(expr[2], x))
    elif expr == 'x':
        return x
    else:
        return expr

# Mutation
def mutate(individual, depth):
    if random.random() < 0.1:
        return generate_random_expression(depth)
    elif isinstance(individual, list):
        if len(individual) == 2:
            return [individual[0], mutate(individual[1], depth - 1)]
        elif len(individual) == 3:
            return [individual[0], mutate(individual[1], depth - 1), mutate(individual[2], depth - 1)]
    return individual

def main():
    population_size = 100
    generations = 40
    tournament_size = 5
    crossover_rate = 0.9
    mutation_rate = 0.1
    max_depth = 5

    points = np.linspace(-5, 5, 100)
    run_no = 5

    for run in range(run_no):
        # Initialize the population
        population = [generate_random_expression(max_depth) for _ in range(population_size)]
        best_individual = None
        best_fitness = float('inf')

        # for generation in range(generations):
        #     fitnessses = [fitness(individual, points) for individual in population]    


if __name__ == "__main__":
    main()
