#Demanda dkji [grupo[dose[unidade]]]
import random
import numpy
import copy
import matplotlib.pyplot as plt
from timeit import default_timer as timer


demanda = [
           [[15, 12], [3, 4],  [3, 4],  [0, 3], [0, 3]],  #G1 indice0
           [[12, 2],  [6, 10], [6, 10], [1, 1], [1, 1]],  #G2 indice1
           [[10, 13], [5, 5],  [5, 5],  [3, 3], [3, 3]],  #G3 indice2
           [[20, 12], [8, 12], [8, 12], [7, 5], [7, 5]]   #G4 indice3
           ]  

pesos = [
         [0.28, 0.34, 0.34, 0.46, 0.46],  #G1 indice0
         [0.24, 0.30, 0.30, 0.42, 0.42],  #G2 indice1
         [0.20, 0.26, 0.26, 0.38, 0.38],  #G3 indice2
         [0.16, 0.22, 0.22, 0.34, 0.34]   #G4 indice3
        ]

disponivel_coronavac = 75

disponivel_astrazeneca = 75

num_grupos = len(demanda)
num_unidades = len(demanda[0][0])

def get_random_alelo(grupo, dose, unidade, left = None):

    if left == None:
        if demanda[grupo][dose][unidade] == 0:
            return 0 
        else:
            return random.randrange(0, demanda[grupo][dose][unidade])
    else:
        safe_range = left if left < demanda[grupo][dose][unidade] else demanda[grupo][dose][unidade]
        if safe_range == 0:
            return 0 
        else:
            return random.randrange(0, safe_range)
 
def generate_solution():

    def randomly(seq):
        shuffled = list(seq)
        random.shuffle(shuffled)
        return iter(shuffled)


    coronavac_left = int(disponivel_coronavac * 0.9)
    astrazeneca_left = int(disponivel_astrazeneca * 0.9)

    solution = []
    for k in range(num_grupos):
        solution.append([])
        for j in range(6):
            solution[k].append([])
            for i in range(num_unidades):
                solution[k][j].append(-1)

    for k in randomly(range(num_grupos)):
        for j in randomly(range(6)):
            j_converted = 0 if j < 2 else j-1
            for i in randomly(range(num_unidades)):
                this_demand = demanda[k][j_converted][i] + 1
                if j == 0:
                    allocated_in_other_dose_1 =  solution[k][1][i]
                    if allocated_in_other_dose_1 != -1:
                        left_from_this_demand = this_demand - allocated_in_other_dose_1
                        if left_from_this_demand < coronavac_left:
                            alocadas = random.randrange(0, left_from_this_demand)
                        else:
                            alocadas = random.randrange(0, coronavac_left)
                    else:
                        alocadas = get_random_alelo(k, j_converted,  i, coronavac_left)
                    coronavac_left -= alocadas
                    solution[k][j][i] = alocadas
                elif j == 1:
                    allocated_in_other_dose_1 =  solution[k][0][i]
                    if allocated_in_other_dose_1 != -1:
                        left_from_this_demand = this_demand - allocated_in_other_dose_1
                        if left_from_this_demand < astrazeneca_left:
                            alocadas = random.randrange(0, left_from_this_demand)
                        else:
                            alocadas = random.randrange(0, astrazeneca_left)
                    else:
                        alocadas = get_random_alelo(k, j_converted,  i, astrazeneca_left)
                    astrazeneca_left -= alocadas
                    solution[k][j][i] = alocadas
                else:
                    if j == 0 or j == 2 or j == 6:
                        alocadas = get_random_alelo(k, j_converted,  i, coronavac_left)
                        coronavac_left -= alocadas
                    else:
                        alocadas = get_random_alelo(k, j_converted,  i, astrazeneca_left)
                        astrazeneca_left -= alocadas
                    solution[k][j][i] = alocadas
    return numpy.array(solution)

def is_feasible(solution):

    # Checar se D1A + D1C < D1
    for k in range(num_grupos):
        for i in range(num_unidades):
            if solution[k][0][i] + solution[k][1][i] > demanda[k][0][i]:
                return False

    # Checar se todas coronovac alocadas nao passam da disponibilidade
    atribuidas_coronavac = 0
    for k in range(num_grupos):
        for j in range(0, 6, 2):
            for i in range(num_unidades):
                atribuidas_coronavac += solution[k][j][i]
    if atribuidas_coronavac > disponivel_coronavac:
        # print("atribuidas_coronavac", atribuidas_coronavac)
        return False

    # Checar se todas coronovac alocadas nao passam da disponibilidade
    atribuidas_astrazeneca = 0
    for k in range(num_grupos):
        for j in range(1, 6, 2):
            for i in range(num_unidades):
                atribuidas_astrazeneca += solution[k][j][i]
    if atribuidas_astrazeneca > disponivel_astrazeneca:
        # print("atribuidas_astrazeneca", atribuidas_astrazeneca, disponivel_astrazeneca)
        return False

    return True

def get_Z(solution):

    accumulated_Z = 0

    for k in range(num_grupos):
            for j in range(6):
                for i in range(num_unidades):
                    j_converted = 0 if j < 2 else j-1
                    accumulated_Z += pesos[k][j_converted] * solution[k][j][i]

    return accumulated_Z

def generate_inital_population(population_size):
    population = []
    while(len(population) < population_size):
        solution = generate_solution()
        if is_feasible(solution) == True:
            # print(len(population)/population_size, end="\r")
            population.append(solution)
    return population

def evaluate_population(population):
    fitness = []
    for solution in population:
        Z = get_Z(solution)
        fitness.append(Z)
    return fitness

def selection(population, fitness):

    def select_solution(population, fitness):
        
        total_fitness = numpy.sum(fitness)
        probabilities = fitness/total_fitness

        choice = random.uniform(0,1)
        index_choice = -1

        for i in range(len(probabilities)):
            interval_start = numpy.sum(probabilities[:i])
            interval_end = interval_start + probabilities[i]
            if interval_start <= choice and choice <= interval_end:
                index_choice = i
        
        chosen_solution = population[index_choice]
        remaining_population = numpy.delete(population, index_choice, axis=0)
        remaining_fitness = numpy.delete(fitness, index_choice, axis=0)

        return (chosen_solution, remaining_population, remaining_fitness)

    population = numpy.array(population)
    fitness = numpy.array(fitness)

    sorted_index = numpy.argsort(fitness)

    sorted_population = population[sorted_index]
    sorted_fitness = fitness[sorted_index]

    remaining_population = sorted_population
    remaining_fitness = sorted_fitness
    pairs = []

    while len(remaining_population) > 0:
        chosen_solution_1, remaining_population, remaining_fitness = select_solution(remaining_population, remaining_fitness)
        chosen_solution_2, remaining_population, remaining_fitness = select_solution(remaining_population, remaining_fitness)
        pairs.append((chosen_solution_1, chosen_solution_2))

    return pairs

def crossover(parents):

    gene_num = 48

    children = []

    for couple in parents:
        children_are_feasible = False

        total_gene_num = 6 * num_grupos * num_unidades

        parent_1 = couple[0].reshape(total_gene_num)
        parent_2 = couple[1].reshape(total_gene_num)

        while not(children_are_feasible):
            split_index = random.randrange(0, gene_num)

            child_1 = numpy.concatenate((parent_1[:split_index], parent_2[split_index:]))
            child_2 = numpy.concatenate((parent_2[:split_index], parent_1[split_index:]))

            child_1 = child_1.reshape((num_grupos, 6, num_unidades))
            child_2 = child_2.reshape((num_grupos, 6, num_unidades))

            if is_feasible(child_1) and is_feasible(child_2):
                children.append(child_1)
                children.append(child_2)
                children_are_feasible = True
    
    return children

def mutation(children, mutation_probability):

    mutated_children = []

    for child in children:
        if random.uniform(0, 1) <= mutation_probability:
            
            new_child_is_feasible = False

            while not new_child_is_feasible:
                
                aux_child = copy.deepcopy(child)

                mutated_gene_k = random.randrange(0, num_grupos)
                mutated_gene_j = random.randrange(0, 6)
                mutated_gene_i = random.randrange(0, num_unidades)
                j_converted = 0 if mutated_gene_j < 2 else mutated_gene_j-1

                alelo = get_random_alelo(mutated_gene_k, j_converted, mutated_gene_i)

                aux_child[mutated_gene_k][mutated_gene_j][mutated_gene_i] = alelo

                new_child_is_feasible = is_feasible(aux_child)

            mutated_children.append(aux_child)
        else:
            mutated_children.append(copy.deepcopy(child))
    
    return mutated_children

def reinsertion(population, fitness, final_size):

    population = numpy.array(population, dtype='object')
    fitness = numpy.array(fitness)

    sorted_index = numpy.argsort(fitness)

    sorted_population = population[sorted_index]

    return (sorted_population[-final_size:])

def get_mean_Z(population):

    sum_of_Z = 0

    for solution in population:
        sum_of_Z += get_Z(solution)

    return sum_of_Z/len(population)

def get_distribution(solution):

    distribution = [0, 0, 0, 0, 0, 0]

    for k in range(num_grupos):
        for j in range(6):
            for i in range(num_unidades):
                distribution[j] += solution[k][j][i]

    return distribution

num_generations = 40
mean_Zs = []

start = timer()

# aleatorias e factiveis
population = generate_inital_population(500)

for gen in range(num_generations):

    fitness = evaluate_population(population)

    parents = selection(population, fitness)

    children = crossover(parents)
    
    children = mutation(children, 0.9)

    intermediate_population = numpy.concatenate((population, children))

    intermediate_population_fitness = evaluate_population(intermediate_population)

    new_population = reinsertion(intermediate_population, intermediate_population_fitness, len(population))

    population = copy.deepcopy(new_population)

    mean_Zs.append(get_mean_Z(population))

    # print(gen)

end = timer()
print(end - start)

plt.plot(range(num_generations), mean_Zs)

print(mean_Zs[-1])
print(get_distribution(population[0]))
  
plt.xlabel('Geracao')
plt.ylabel('Z medio')
plt.show()