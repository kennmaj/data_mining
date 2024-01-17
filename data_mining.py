import numpy as np
import time
import csv

def minkosky_distance(m):
    def distance(p, q):
        return np.linalg.norm(np.array(p) - np.array(q), ord = m)
    return distance

def calculate_distance_to(r, distance_fun):
    def map_function(q):
        return [*q, distance_fun(q[1], r)]
    return map_function
    
def sort_based_on_distance(x): return x[2]

def sort_D(D, r, distance):
    return list(sorted(map(calculate_distance_to(r, distance),  D), key = sort_based_on_distance))

def eps_neighborhood(D, m, eps):

    ids = list(map(lambda x : x[0], D))

    distance = minkosky_distance(m)

    out_list = []

    for p in ids :
        p_value = D[ids.index(p)]

        D_copy = D.copy()
        D_copy.remove(p_value)
        D_sorted = sort_D(D_copy, p_value[1], distance)

        out_list.append([p,  list(filter(lambda x : x[2] <= eps , D_sorted)), len(D_copy), p_value[1] ])

    return out_list

def ti_backward_nieghborhood(D, p, p_dist, eps, distance):
        seeds = []
        backward_threshold = p_dist - eps
        dist_cal = 0
        for i in range(len(D) - 1, -1, - 1):
            q = D[i]
            if q[2] < backward_threshold:
                break
            # else : 
            #     seeds.append(q)
            
            dist_val = distance(q[1], p[1]) 
            dist_cal += 1
            if distance(q[1], p[1]) <= eps:
                seeds.append([*q[:2], dist_val])
        return seeds, dist_cal


def ti_forward_neighborhood(D, p, p_dist, eps, distance):
    seeds = []
    forward_threshold = eps + p_dist
    dist_cal = 0
    for i in range(len(D)):
        q = D[i]
        if q[2] > forward_threshold:
            break
        # else : 
        #     seeds.append(q)
        dist_val = distance(q[1], p[1]) 
        dist_cal += 1
        if dist_val <= eps:
            seeds.append([*q[:2], dist_val])
    return seeds, dist_cal

def ti_neighborhood(D_sorted, p, eps, distance):

    p_index = list(map(lambda x : x[0], D_sorted)).index(p)
    p_dist = D_sorted[p_index][2]
 
    backward_neighbors, dist_cal_backward = ti_backward_nieghborhood(D_sorted[:p_index].copy(), D_sorted[p_index], p_dist, eps, distance)
    forward_neighbors, dist_cal_forward = ti_forward_neighborhood(D_sorted[p_index + 1:].copy(), D_sorted[p_index], p_dist, eps, distance)

    
    return backward_neighbors + forward_neighbors, dist_cal_forward + dist_cal_backward, D_sorted[p_index][1] 


def eps_ti_neighborhood(D, r, m, eps):
    
    distance = minkosky_distance(m)

    D_sorted = sort_D(D, r, distance)

    out_list = []

    nr_of_dist_cal = len(D)

    for p in list(map(lambda x : x[0], D_sorted)):
        out_set, nr_of_dist_cal_for_p, p_dim = ti_neighborhood(D_sorted.copy(), p, eps, distance)
        out_list.append([p, out_set, nr_of_dist_cal + nr_of_dist_cal_for_p, p_dim])

    return out_list

def print_return(out_list):
    for val in out_list:
        # if val[0] != "F":
        #     continue
        print(f"ID: {val[0]}")
        if val[1] == []:
            print("\tEMPTY")
            continue

        for ans in val[1]:
            print(f"\t{ans}")

def prepare_alg_out(out_list):
    alg_out = []
    for val in out_list:
        alg_out.append([val[0], *val[3], val[2] , len(val[1]), list(map(lambda x: x[0], val[1]))])

    return alg_out



def print_to_file(algorithm_type, file_parameters, alg_paremteters, alg_out):




    out_file_path = f"out_data/{file_parameters['fname']}/OUT_{algorithm_type}_{file_parameters['fname']}_D{len(file_parameters['dimensions'])}_R{file_parameters['rows']}_m{alg_paremteters['m']}_Eps{alg_paremteters['Eps']}.csv"
    stat_file_path = f"out_data/{file_parameters['fname']}/STAT_{algorithm_type}_{file_parameters['fname']}_D{len(file_parameters['dimensions'])}_R{file_parameters['rows']}_m{alg_paremteters['m']}_Eps{alg_paremteters['Eps']}.csv"


    with open(out_file_path, mode='w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(["point_id", *[f"v{val}" for val in list(map(int, file_parameters['dimensions']))], "#_of_distance_calculations", "|N_Eps|", "ids_of_points_in_N_Eps"])

        for val in alg_out:
            writer.writerow(val)

# partial runtimes for each important phase of the algorithm such as: 
# reading the input file, determining min and max values for each dimension, 
# calculating distances from each point in the input file to all reference vectors, 
# (physical or logical) sorting of all points with respect to reference vector 1, calculating Eps-neigbourhoods, saving results to output files
        
    with open(stat_file_path, mode='w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow([
            "name_of_the_input_file", 
            "#_of_dimensions_of_a_point", 
            "#_of_points_in_the_input_file", 
            "min_values_for_all_dimensions", 
            "max_values_for_all_dimensions",  
            "value_of_parameter_Eps", 
            "value_of_parameter_m", 
            "the_number_of_used_reference_vectors", 
            "reference_vector_1",
            "!!! PARTIAL RUNTIMES !!!", 
            "total_time", 
            "#_of_distance_calculations_between_points_in_the_input_file_and_reference_vectors"
            "least_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
            "greatest_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
            "avg_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
            "variance_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
            "least_cardinalities_of_determined_Eps-neighbourhoods"
            "greatest_cardinalities_of_determined_Eps-neighbourhoods"
            "avg_cardinalities_of_determined_Eps-neighbourhoods"
            "variance_cardinalities_of_determined_Eps-neighbourhoods"
              ])



# D = [
#     ["F", [1.1, 3.0]],
#     ["C", [2.8, 3.5]],
#     ["A", [4.2, 4.0]],
#     ["K", [0.9, 0.0]],
#     ["L", [1.0, 1.5]],
#     ["G", [0.0, 2.4]],
#     ["H", [2.4, 2.0]],
#     ["B", [5.9, 3.9]]
# ]

def datafile_and_transform(filename):
    def get_data(filename):
        transform_line = lambda x : [x[0], list(map(float, x[1:]))]
        with open(filename) as f:
            return [transform_line(line.strip().split(",")) for line in f]
    data = get_data(filename)
        
    return data[0], data[1:]

input_fname = "toy_dataset"

input_filepath = f"input_data/data/{input_fname}.csv"
file_column_name, D = datafile_and_transform(input_filepath)


file_parameters = {
    "fname" : input_fname,
    "dimensions" : file_column_name[1],
    "rows" : len(D)

}

alg_paremteters = {
    "r" : [0,0],
    "m" : 2,
    "Eps" : 1.5
}


r = [0,0]
m = 2
eps = 1.5

print("EPS-NB")
now = time.time_ns()

out_list = eps_neighborhood(D.copy(), m, eps = eps)

alg_out = prepare_alg_out(
    out_list = out_list
)


print_to_file(
    algorithm_type = "EPS-NB", 
    file_parameters = file_parameters, 
    alg_paremteters= alg_paremteters,
    alg_out = alg_out
)


print(time.time_ns() - now)

print("EPS-TI-NB")
now = time.time_ns()


out_list = eps_ti_neighborhood(D.copy(), r, m, eps = eps)

alg_out = prepare_alg_out(
    out_list = out_list
)

print_to_file(
    algorithm_type = "EPS-TI-NB", 
    file_parameters = file_parameters, 
    alg_paremteters= alg_paremteters,
    alg_out = alg_out
)


print(time.time_ns() - now)


