#Demanda dkji [grupo[dose[unidade]]]
import random
import numpy
import copy
import matplotlib.pyplot as plt
from timeit import default_timer as timer


demanda = [
    [[0,0,0,0,0,0,0,0,0,0], [1727,929,243,1186,10455,4838,1972,17395,6639,8851], [2289,1884,372,752,27918,10360,1781,61895,38832,41525], [0,154,4,34,202,426,19,953,730,3559], [1728,1108,223,672,17329,6588,1656,38053,29297,42681]],#GP1
    [[1557,8074,9444,1537,36774,17193,6594,52382,40101,116365], [9644,27195,55348,319,111304,18857,18220,151458,42545,560430], [4530,39496,10327,12073,220391,82023,46448,437880,373380,697965], [10,41,872,8,254,82,85,470,145,1838], [2568,25803,7126,4432,112324,28852,31075,149280,78664,267589]],#GP10
    [[16514,85643,100174,16306,390079,182377,69944,555638,425369,1234329], [6488,31648,22011,2981,187450,37576,35399,249773,339222,135972], [10217,58195,62620,10273,169107,12559,2845,57883,120916,122865], [11,134,652,11,697,113,163,724,1148,1244], [3355,16547,51910,1789,17932,2116,1031,7581,5513,34117]],#GP11
    [[65808,341283,399189,64977,1554441,726761,278722,2214183,1695067,4918723], [50,13610,4487,2132,7671,151,91,835,6385,0], [459,1781,69684,1813,6398,246,601,1206,1539,5], [0,63,63,5,45,43,29,3,47,0], [52,100,18325,39,2496,119,476,896,631,11]],#GP12
    [[18819,97596,114155,18581,444519,207830,79705,633183,484733,1406593], [6,2,81,12,297,123,12,212,1071,2], [16,2,63,20,252,8,148,348,2644,6], [0,0,1,0,6,85,0,0,66,0], [13,1,6,11,160,7,131,257,1828,7]],#GP13
    [[393,2038,2384,388,9282,4340,1664,13221,10121,29370], [0,0,0,0,4,0,1,2,20,277], [0,0,0,0,6,5,13,3,36,6424], [0,0,0,0,0,0,0,0,10,7], [0,0,0,0,6,2,4,3,25,2057]],#GP14
    [[2541,13176,15411,2509,60011,28058,10760,85481,65440,189894], [48,55,24,0,179,0,57,650,2374,1], [377,280,41,30,54,10,436,1674,848,1], [0,0,0,0,1,0,0,3,1,0], [3,109,10,0,21,15,40,135,112,2]],#GP15
    [[7332,38021,44473,7239,173176,80967,31052,246676,188843,547982], [1,530,463,1,15673,2,7,106,352,26206], [3,833,3749,3,242,6,67,18,382,179913], [0,1,0,0,35,0,0,0,30,385], [0,4,2660,2,26,6,57,11,316,4416]],#GP16
    [[2132,11057,12933,2105,50362,23546,9030,71736,54918,159359], [0,1,3,0,2161,0,7,107,57,5], [2,28,776,1,65,9,145,15,235,54], [0,0,2,0,6,0,1,3,13,1], [1,2,538,0,34,1,132,16,179,2]],#GP17
    [[2227,11551,13511,2199,52613,24599,9434,74943,57373,166483], [164,542,2251,845,6575,2438,955,4281,21239,241], [1149,3056,6218,1082,4383,2309,5783,10446,3945,12], [0,4,6,16,93,4,9,20,48,4], [105,1149,2095,110,977,192,1152,2356,964,16]],#GP18
    [[2027,10514,12298,2002,47890,22390,8587,68215,52222,151538], [0,0,2,0,1598,0,4,32,4,0], [2,2,0,0,46,31,4,17,57,0], [0,0,0,0,3,0,1,1,1,0], [2,1,1,0,20,26,2,9,7,0]],#GP19
    [[0,0,0,0,0,0,0,0,0,0], [0,2,4,0,215,2,2,314,415,2534], [15,30,53,80,439,293,195,1785,2029,4522], [0,1,0,6,29,0,0,31,80,572], [13,17,49,73,301,135,190,1616,1800,3363]],#GP2
    [[221,1145,1339,218,5214,2438,935,7427,5686,16499], [0,0,0,0,2,0,3,0,0,0], [0,0,2,0,1,1,6,1,6,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,4,3,1,0,2,9]],#GP20
    [[350,1814,2122,345,8264,3864,1482,11772,9012,26151], [0,0,0,0,18,0,0,0,3,0], [0,0,4,0,6,0,3,1,10,0], [0,0,0,0,0,0,0,0,1,0], [0,0,4,0,7,0,1,0,1,5]],#GP21
    [[125,646,756,123,2942,1375,528,4191,3208,9309], [0,0,16,2,0,0,0,1,0,1], [3,2,13,0,1,0,15,2,1,1], [0,0,2,0,0,0,0,0,0,0], [3,0,7,0,0,0,5,1,0,0]],#GP22
    [[3730,19345,22627,3683,88109,41194,15799,125504,96080,278803], [0,0,1,4,12,1,1,11,7,0], [1,1,0,0,9,0,1,11,15,0], [0,0,0,0,0,0,0,0,2,0], [1,1,1,0,7,0,1,8,5,0]],#GP23
    [[334,1731,2025,330,7886,3687,1414,11233,8599,24953], [0,0,1,0,29,2,0,2,7,0], [3,0,10,0,6,0,2,7,4,0], [0,0,0,0,0,0,0,0,0,0], [1,0,7,0,6,0,0,7,1,0]],#GP24
    [[16002,82989,97070,15800,377991,176726,67777,538420,412187,1196080], [0,0,1,39,17,0,0,17,14,0], [1,0,1,0,4,2,3,12,56,1], [0,0,0,0,1,0,0,4,1,0], [0,1,0,0,4,0,3,6,22,0]],#GP25
    [[0,0,0,0,0,0,0,0,0,0], [0,253,135,0,42,2,58,20,443,299], [7019,7261,59945,4180,17275,18423,125,6926,12159,3616], [0,0,4,0,0,5,3,0,4,15], [4329,6530,35415,3452,12532,17180,50,6359,9777,3243]],#GP3
    [[0,0,0,0,0,0,0,0,0,0], [5888,7380,9336,6930,185218,66841,15231,190203,127765,506775], [13867,73325,84256,11923,209578,288792,89498,355612,301314,1019475], [774,215,1649,2429,35226,6932,4647,19508,45101,69825], [10614,64682,73740,10297,173221,222163,70393,285903,228277,888575]],#GP4
    [[0,0,0,0,0,0,0,0,0,0], [5438,12815,25744,485,19414,30587,22412,14387,29565,102801], [1344,29442,2423,5282,209427,91384,15939,332526,200951,699361], [20,106,11075,12,219,2054,126,81,177,945], [975,26106,1596,4338,167028,55664,13700,276619,171247,606440]],#GP5
    [[0,0,0,0,0,0,0,0,0,0], [5438,12815,25744,485,19414,30587,22412,14387,29565,102801], [1344,29442,2423,5282,209427,91384,15939,332526,200951,699361], [20,106,11075,12,219,2054,126,81,177,945], [975,26106,1596,4338,167028,55664,13700,276619,171247,606440]],#GP6
    [[721,3738,4372,712,17024,7959,3052,24249,18564,53868], [2463,0,29349,2265,41,0,0,2463,0,0], [3799,0,2425,191,0,0,0,130,0,0], [2,0,48,4,1,0,0,3,0,0], [492,0,309,86,5,1,0,2,0,0]],#GP7
    [[2402,12456,14570,2372,56735,26526,10173,80814,61867,179526], [0,4747,1064,3146,56520,267,0,29192,8250,1813], [0,1,15,360,6696,38,2,282,1185,5280], [0,11,0,105,269,1,0,26,51,25], [0,0,0,109,2046,0,2,104,1115,4241]],#GP8
    [[0,0,0,0,0,0,0,0,0,0], [4283,10571,38503,195,60852,2674,1415,10252,10938,27885], [6051,51300,6065,8013,246547,135478,54865,482701,333067,1106997], [20,45,12192,8,172,119,40,88,333,603], [5098,41263,4240,6140,173471,58516,42222,348649,235910,860253]]#GP9
           ]  

pesos = [
	 [0.083, 0.138,	 0.192,	 0.138,	 0.192],   #GP1	
	 [0.082, 0.137,	 0.191,	 0.137,	 0.191],   #GP2	
	 [0.081, 0.136,	 0.190,	 0.136,	 0.190],   #GP3	
	 [0.080, 0.134,	 0.189,	 0.134,	 0.189],   #GP4	
	 [0.079, 0.133,	 0.188,	 0.133,	 0.188],   #GP5	
	 [0.078, 0.132,	 0.187,	 0.132,	 0.187],   #GP6	
	 [0.077, 0.131,	 0.186,	 0.131,	 0.186],   #GP7	
	 [0.076, 0.130,	 0.185,	 0.130,	 0.185],   #GP8	
	 [0.075, 0.129,	 0.184,	 0.129,	 0.184],   #GP9	
	 [0.074, 0.128,	 0.183,	 0.128,	 0.183],   #GP10
	 [0.073, 0.127,	 0.182,	 0.127,	 0.182],   #GP11
	 [0.071, 0.126,	 0.181,	 0.126,	 0.181],   #GP12
	 [0.070, 0.125,	 0.180,	 0.125,	 0.180],   #GP13
	 [0.069, 0.124,	 0.178,	 0.124,	 0.178],   #GP14
	 [0.068, 0.123,	 0.177,	 0.123,	 0.177],   #GP15
	 [0.067, 0.122,	 0.176,	 0.122,	 0.176],   #GP16
	 [0.066, 0.121,	 0.175,	 0.121,	 0.175],   #GP17
	 [0.065, 0.120,	 0.174,	 0.120,	 0.174],   #GP18
	 [0.064, 0.119,	 0.173,	 0.119,	 0.173],   #GP19
	 [0.063, 0.118,	 0.172,	 0.118,	 0.172],   #GP20
	 [0.062, 0.116,	 0.171,	 0.116,	 0.171],   #GP21
	 [0.061, 0.115,	 0.170,	 0.115,	 0.170],   #GP22
	 [0.060, 0.114,	 0.169,	 0.114,	 0.169],   #GP23
	 [0.059, 0.113,	 0.168,	 0.113,	 0.168],    #GP24
	 [0.058, 0.112,	 0.167,	 0.112,  0.167]     #GP25	 
        ]

disponivel_coronavac = 10000000
disponivel_astrazeneca = 10000000

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