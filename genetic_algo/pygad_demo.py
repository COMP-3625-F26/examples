import numpy as np
import pygad

def himmelblaus_function(input_vector) -> float:
    """
    a classic function used to test the performance of local search algorithms
    :param input_vector: vector of [x, y] values
    :return: function value at (x, y)
    """
    x, y = input_vector
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def fitness_function(ga_instance, solution, solution_idx):
    return -himmelblaus_function(solution)

# custom selection operator (random selection)
def random_selection(fitness, num_parents, ga_instance):
    selected_idx = np.random.choice(len(fitness), size=num_parents, replace=False)
    return ga_instance.population[selected_idx], selected_idx

ga = pygad.pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    sol_per_pop=25,
    fitness_func=fitness_function,
    init_range_high=5,
    init_range_low=-5,
    num_genes=2,
    parent_selection_type=random_selection,
    crossover_type='single_point',
    mutation_type='random',
    mutation_percent_genes=0.1,
)

ga.run()
solution, solution_fitness, solution_idx = ga.best_solution()
print(solution, solution_fitness)