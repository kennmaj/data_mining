#include "Eigen/Dense"
#include "fmt/format.h"
#include <chrono>
#include <map>
#include <filesystem>
#include <ranges>

using namespace std::chrono_literals;

using Dataset = std::vector<Eigen::MatrixXd>;

template<int order> double minkosky_distance(Eigen::MatrixXd p, Eigen::MatrixXd q)
{
    return (p - q).lpNorm<order>();
}

double calculate_distance_to(double r, std::function<double(double,double)>distance_fun)
{
    auto map_function = [](Eigen::MatrixXd q)
    {
        return [ *q, distance_fun(q[1], r) ];
    };
    return map_function;
}

double sort_based_on_distance(Eigen::MatrixXd  x)
{
    return x[2];
}

void sort_D(Dataset D, Eigen::MatrixXd  r, double distance)
{
    std::ranges::sort(D,[r](Eigen::MatrixXd ){ calculate_distance_to(r, distance);  });
    return list(sorted(std::ranges::transform(D, calculate_distance_to(r, distance)), key = sort_based_on_distance));
}

void eps_neighborhood(std::vector<Eigen::MatrixXd> D, int m, double eps)
{
    std::vector<double> ids;
    std::ranges::transform(D,ids,[](Eigen::MatrixXd x){return x[0];});
    auto distance = minkosky_distance(m);
    auto out_list = [];
    auto time_to_calc_all_ref = 0ms;
    for (auto p : ids)
    {
        auto p_value = D[ids.index(p)];
        auto time_now = std::chrono::steady_clock::now();
        auto D_copy = D;
        D_copy.remove(p_value);
        auto D_sorted = sort_D(D_copy, p_value[1], distance);
        time_to_calc_all_ref += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_now);
        out_list.append([ p, list(std::ranges::views::filter(D_sorted,[&eps](auto x){return x[2] <= eps;})),len(D_copy),p_value[1] ]);
    }
    return out_list, time_to_calc_all_ref;
}

void ti_backward_nieghborhood(Dataset D, Eigen::MatrixXd p, double p_dist, double eps,std::function<double(double,double)> distance)
{
    seeds = [];
    backward_threshold = p_dist - eps;
    dist_cal = 0;
    for (auto i : range(len(D) - 1, -1, -1))
    {
        auto q = D[i];
        if (q[2] < backward_threshold)
        {
            break;
        }
        //      else {
        //          seeds.append(q)
        //      }
        dist_val = distance(q[1], p[1]);
        dist_cal += 1;
        if (distance(q[1], p[1]) <= eps)
        {
            seeds.append([ *q[:2], dist_val ]);
        }
    }
    return seeds, dist_cal;
}

void ti_forward_neighborhood(D,Eigen::MatrixXd p, double p_dist, double eps,std::function<double(double,double)> distance)
{
    seeds = [];
    forward_threshold = eps + p_dist;
    dist_cal = 0;
    for (auto i; range(len(D)))
    {
        q = D[i];
        if (q[2] > forward_threshold)
        {
            break;
        }
        //      else {
        //          seeds.append(q)
        //      }
        dist_val = distance(q[1], p[1]);
        dist_cal += 1;
        if (dist_val <= eps)
        {
            seeds.append([ *q[:2], dist_val ])
        }
    }

    return seeds, dist_cal;
}

void ti_neighborhood(Dataset D_sorted,Eigen::MatrixXd p, double eps,std::function<double(double,double)> distance)
{
    auto p_index = list(map(lambda x : x[0], D_sorted)).index(p);
    auto p_dist = D_sorted[p_index][2];

    auto [backward_neighbors, dist_cal_backward] = ti_backward_nieghborhood(
        D_sorted[:p_index].copy(), D_sorted[p_index], p_dist, eps, distance);
    auto [forward_neighbors, dist_cal_forward] = ti_forward_neighborhood(
        D_sorted [p_index + 1:].copy(), D_sorted[p_index], p_dist, eps, distance);

    return backward_neighbors + forward_neighbors, dist_cal_forward + dist_cal_backward,
           D_sorted[p_index][1]
}

void eps_ti_neighborhood(Dataset D, double r, int m, double eps)
{
    auto distance = minkosky_distance(m);

    auto time_now = std::chrono::steady_clock::now();
    D_sorted = sort_D(D, r, distance);
    time_to_calc_all_ref = time.time_ns() - time_now;

    out_list = [];

    auto nr_of_dist_cal = len(D);

    for (auto p : list(map(lambda x : x[0], D_sorted)))
    {
        auto [out_set, nr_of_dist_cal_for_p, p_dim] =
            ti_neighborhood(D_sorted.copy(), p, eps, distance);
        out_list.append([ p, out_set, nr_of_dist_cal + nr_of_dist_cal_for_p, p_dim ]);
    }
    return out_list, time_to_calc_all_ref;
}

void print_return(out_list)
{
    for (val : out_list)
    {
        fmt::print("ID: {}",val[0]);
        if (val[1] == [])
        {
            fmt::print("\tEMPTY");
            continue;
        }
        for (ans : val[1])
        {
            fmt::print(f "\t{}",ans);
        }
    }
}

void prepare_alg_out(out_list)
{
    alg_out = [];
    for(val : out_list)
    {
        alg_out.append([ val[0], *val[3], val[2], len(val[1]), list(map(lambda x : x[0], val[1])) ]);
    }
return alg_out;
}

void print_to_file(algorithm_type, file_parameters, alg_paremteters, alg_out, time_out){


    out_file_path = fmt::format("out_data/{}/OUT_{}_{}_D{}_R{}_m{}_Eps{}.csv",file_parameters['fname'],algorithm_type,file_parameters['fname'],len(file_parameters['dimensions']),file_parameters['rows'],alg_paremteters['m'],alg_paremteters['Eps']);
    stat_file_path = fmt::format("out_data/{file_parameters['fname']}/STAT_{algorithm_type}_{file_parameters['fname']}_D{len(file_parameters['dimensions'])}_R{file_parameters['rows']}_m{alg_paremteters['m']}_Eps{alg_paremteters['Eps']}.csv");

    with open(out_file_path, mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow([
    "point_id",
    *[f"v{val}" for val in list(map(int, file_parameters['dimensions']))],
    "#_of_distance_calculations",
    "|N_Eps|",
    "ids_of_points_in_N_Eps"
    ])

    for val in alg_out:
    writer.writerow(val)

    with open(stat_file_path, mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow([
    "name_of_the_input_file",
    "#_of_dimensions_of_a_point",
    "#_of_points_in_the_input_file",
    *[f"min_v{val}" for val in list(map(int, file_parameters['dimensions']))],
    *[f"max_v{val}" for val in list(map(int, file_parameters['dimensions']))],
    "value_of_parameter_Eps",
    "value_of_parameter_m",
    "the_number_of_used_reference_vectors",
    "reference_vector_1",
    "reading_the_input_file",
    "determining_min_and_max_values_for_each_dimension",
    "time_calculating_distances_from_each_point_in_the_input_file_to_all_reference_vectors"
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

    dist_lst = [val[3] for val in alg_out]
    card_lst = [val[3] for val in alg_out]


    writer.writerow([
    file_parameters['fname'],
    len(file_parameters['dimensions']),
        file_parameters['rows'],
        *file_parameters["min_values_for_all_dimensions"],
        *file_parameters["max_values_for_all_dimensions"],

        alg_paremteters['Eps'],
        alg_paremteters['m'],
    0 if algorithm_type == "EPS-NB" else 1,
    alg_paremteters['r'],

    time_out["reading_the_input_file"],
    time_out["determining_min_and_max_values_for_each_dimension"],
    time_out["time_calculating_distances_from_each_point_in_the_input_file_to_all_reference_vectors"],
    time_out["total_time"],

    sum(dist_lst),
        min(dist_lst),
        max(dist_lst),
        np.average(dist_lst),
        np.var(dist_lst),

        min(card_lst),
        max(card_lst),
        np.average(card_lst),
        np.var(card_lst)

    ])
}
// D = [
// ["F", [1.1, 3.0]],
// ["C", [2.8, 3.5]],
// ["A", [4.2, 4.0]],
// ["K", [0.9, 0.0]],
// ["L", [1.0, 1.5]],
// ["G", [0.0, 2.4]],
// ["H", [2.4, 2.0]],
// ["B", [5.9, 3.9]]
// ]

void datafile_and_transform(std::filesystem::path filename)
{
    void get_data(filename)
    {
        transform_line = lambda x : [ x[0], list(map(float, x [1:])) ];
        with open(filename) as f
        {
            return [transform_line(line.strip().split(",")) for line in f];
        }
    }
    data = get_data(filename);

    return data[0], data [1:];
}

void determine_min_and_max_values_for_each_dimension(Dataset D)
{
    min_values = D[0][1].copy();
    max_values = D[0][1].copy();
    for (auto val : D [1:])
    {
        for (auto [id_vv, vv] : enumerate(val[1]))
        {
            if (min_values[id_vv] > vv)
            {
                min_values[id_vv] = vv;
            }
            if (max_values[id_vv] < vv)
            {
                max_values[id_vv] = vv;
            }
        }
    }
    return min_values, max_values;
}

int main()
{
    auto input_fname = std::string{"toy_dataset"};
    auto input_filepath = fmt::format("input_data/data/{}.csv", input_fname);
    auto time_now = std::chrono::steady_clock::now();
    auto [file_column_name, D] = datafile_and_transform(input_filepath);
    auto reading_the_input_file = std::chrono::steady_clock::now() - time_now;
    time_now = std::chrono::steady_clock::now();
    auto [min_values_for_all_dimensions, max_values_for_all_dimensions] =
        determine_min_and_max_values_for_each_dimension(D);
    auto determining_min_and_max_values_for_each_dimension =
        std::chrono::steady_clock::now() - time_now;

    auto file_parameters = std::map<std::string_view, double>{
        {"fname", input_fname},
        {"dimensions", file_column_name[1]},
        {"rows", len(D)},
        {"min_values_for_all_dimensions", min_values_for_all_dimensions},
        {"max_values_for_all_dimensions", max_values_for_all_dimensions}};

    alg_paremteters = {
        "r" : [ 0, 0 ],
        "m" : 2,
        "Eps" : 1.5
    }

    auto r = [ 0, 0 ];
    auto m = 2;
    auto eps = 1.5;

    fmt::print("EPS-NB");

    time_now = std::chrono::steady_clock::now();

    auto [out_list, time_to_calc_all_ref] = eps_neighborhood(D.copy(), m, eps = eps);

    auto total_time = std::chrono::steady_clock::now() - time_now;

    auto alg_out = prepare_alg_out(out_list);

    auto time_out = std::map<std::string_view, std::chrono::milliseconds>{
        {"reading_the_input_file", reading_the_input_file},
        {"determining_min_and_max_values_for_each_dimension",
         determining_min_and_max_values_for_each_dimension},
        {"time_calculating_distances_from_each_point_in_the_input_file_to_all_"
         "reference_vectors",
         time_to_calc_all_ref},
        {"total_time", total_time}};
    print_to_file("EPS-NB", file_parameters, alg_paremteters, alg_out, time_out);

    fmt::print("EPS-TI-NB");

    time_now = std::chrono::steady_clock::now();
    [ out_list, time_to_calc_all_ref ] = eps_ti_neighborhood(D.copy(), r, m, eps = eps);
    total_time = std::chrono::steady_clock::now() - time_now;

    alg_out = prepare_alg_out(out_list);

    time_out = {{"reading_the_input_file", reading_the_input_file},
                {"determining_min_and_max_values_for_each_dimension",
                 determining_min_and_max_values_for_each_dimension},
                {"time_calculating_distances_from_each_point_in_the_input_file_"
                 "to_all_reference_vectors",
                 time_to_calc_all_ref},
                {"total_time", total_time}};

    print_to_file("EPS-TI-NB", file_parameters, alg_paremteters, alg_out, time_out);
}
