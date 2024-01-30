import pandas as pd
import os

round_cols = [
    "value_of_parameter_Eps",
    "avg_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "variance_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "avg_cardinalities_of_determined_Eps-neighbourhoods",
    "variance_cardinalities_of_determined_Eps-neighbourhoods"
]

select_columns = [
    "value_of_parameter_Eps",
    "value_of_parameter_m", 
    "reference_vector_1", 
    "total_time", 
    "#_of_distance_calculations_between_points_in_the_input_file_and_reference_vectors",
    "least_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "greatest_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "avg_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "variance_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point",
    "least_cardinalities_of_determined_Eps-neighbourhoods",
    "greatest_cardinalities_of_determined_Eps-neighbourhoods",
    "avg_cardinalities_of_determined_Eps-neighbourhoods",
    "variance_cardinalities_of_determined_Eps-neighbourhoods"
    ]

for test_case_dir in os.listdir("out_data"):
    print(test_case_dir)

    if test_case_dir == "toy_dataset": continue
    ti_cases = []
    eps_cases = []
    for test_case_file in os.listdir(f"out_data/{test_case_dir}"):

        if test_case_file.startswith("STAT_TI"):
            data = pd.read_csv(f"out_data/{test_case_dir}/{test_case_file}")[select_columns]
            data[round_cols] = data[round_cols].round(2)
            data.columns = ["Eps", "m", "r", "TT", "#DISCAL", "min#DCAL", "max#DCAL", "avg#DCAL", "var#DCAL", "min#CARD", "max#CARD", "avg#CARD", "var#CARD"]
            ti_cases.append(data)

        elif test_case_file.startswith("STAT_EPS"):
            data = pd.read_csv(f"out_data/{test_case_dir}/{test_case_file}")[select_columns]
            data[round_cols] = data[round_cols].round(2)
            data.columns = ["Eps", "m", "r", "TT", "#DISCAL", "min#DCAL", "max#DCAL", "avg#DCAL", "var#DCAL", "min#CARD", "max#CARD", "avg#CARD", "var#CARD"]
            eps_cases.append(data)

    pd.concat(ti_cases).to_csv(f"out_data_analysis_cpp/{test_case_dir}/TI_EPS_NB.csv", index=None)
    pd.concat(eps_cases).to_csv(f"out_data_analysis_cpp/{test_case_dir}/EPS_NB.csv", index=None)


