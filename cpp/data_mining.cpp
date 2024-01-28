#include "Eigen/Dense"
#include "fmt/format.h"
#include <chrono>
#include <map>
#include <filesystem>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/algorithm/find_first_of.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/zip_with.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/to_container.hpp>

#include <string_view>
#include <fstream>
#include <charconv>

using namespace std::chrono_literals;
using namespace std::chrono;
using namespace ranges;
using namespace ranges::view;
using namespace std::literals;

struct Point
{
    int id;
    std::vector<double> values;
    [[nodiscard]] auto minkowskiDistanceTo(Point other, double order) const -> double
    {
        auto pom = zip_with(std::minus<>(), values, other.values) |
                   transform([](double x) { return std::abs(x); }) |
                   transform([order](double x) { return std::pow(x, order); });
        auto sum = accumulate(pom, 0.0, std::plus<>());
        return std::pow(sum, 1.0 / order);
    }
};

struct Dataset
{
    std::vector<Point> data;
};

struct DatasetWithReference
{
    struct Row
    {
        Point point;
        double distance_to_reference;
    };
    std::vector<Row> data;
    Point reference;
};
struct SortedDataset : public DatasetWithReference
{
};

struct Neighborhood : public DatasetWithReference
{
    std::size_t number_of_distance_computations{0};
};

SortedDataset sort_D(Dataset D, const Point &reference_point, double order)
{
    SortedDataset d;
    d.data = D.data |
             transform(
                 [&reference_point, &order](const Point &p)
                 {
                     return SortedDataset::Row{.point = p,
                                               .distance_to_reference =
                                                   p.minkowskiDistanceTo(reference_point, order)};
                 }) |
             to<std::vector>;
    d.reference = reference_point;
    sort(d.data,
         [reference_point](const SortedDataset::Row &row1, const SortedDataset::Row &row2)
         { return row1.distance_to_reference < row2.distance_to_reference; });
    return d;
}

struct NeighborhoodComputation
{
    std::vector<Neighborhood> neighborhoods;
    std::chrono::milliseconds computation_time;
};

NeighborhoodComputation eps_neighborhood(Dataset D, int minkowski_distance_order, double eps)
{
    using namespace std::chrono;
    auto ids = transform(D.data, [](const Point &x) { return x.id; });
    auto out_list = std::vector<Neighborhood>{};
    auto time_to_calc_all_ref = 0ms;
    for (auto p : ids)
    {
        auto find_result = find_if(D.data, [&p](const Point &r) { return r.id == p; });
        if (find_result == D.data.end())
            throw std::logic_error("wrong id!");
        auto p_value = *find_result;
        auto time_now = steady_clock::now();
        auto D_copy = D;
        std::erase_if(D_copy.data, [&p_value](const Point &r) { return r.id == p_value.id; });
        auto D_sorted = sort_D(D_copy, p_value, minkowski_distance_order);
        time_to_calc_all_ref += duration_cast<milliseconds>(steady_clock::now() - time_now);
        Neighborhood n;
        n.reference = p_value;
        n.data = filter(D_sorted.data, [&eps](auto x) { return x.distance_to_reference <= eps; }) |
                 to<std::vector>;
        n.number_of_distance_computations = D_sorted.data.size();
        out_list.emplace_back(n);
    }
    return {.neighborhoods = out_list, .computation_time = time_to_calc_all_ref};
}

struct NeighborhoodCalculationCalls
{
    Neighborhood seeds;
    int distance_calculation_calls;
};

NeighborhoodCalculationCalls ti_backward_nieghborhood(
    SortedDataset D, const Point& p, double p_dist, double eps, double order)
{
    auto seeds = Neighborhood{};
    auto backward_threshold = p_dist - eps;
    auto dist_cal = 0;
    for (auto i = D.data.size() - 1; i != -1; --i)
    {
        auto q = D.data[i];
        if (q.distance_to_reference < backward_threshold)
        {
            break;
        }
        auto dist_val = q.point.minkowskiDistanceTo(p, order);
        dist_cal += 1;
        if (dist_val <= eps)
        {
            Neighborhood::Row r;
            r.point = q.point;
            r.distance_to_reference = dist_val;
            seeds.data.push_back(r);
        }
    }
    return {seeds, dist_cal};
}

NeighborhoodCalculationCalls ti_forward_neighborhood(
    const DatasetWithReference& D, Point p, double p_dist, double eps, double order)
{
    auto seeds = Neighborhood{};
    auto forward_threshold = eps + p_dist;
    auto dist_cal = 0;
    for (const auto &q : D.data)
    {
        if (q.distance_to_reference > forward_threshold)
        {
            break;
        }
        auto dist_val = q.point.minkowskiDistanceTo(p, order);
        dist_cal += 1;
        if (dist_val <= eps)
        {
            auto r = Neighborhood::Row{};
            r.point = q.point;
            r.distance_to_reference = dist_val;
            seeds.data.push_back(r);
        }
    }
    return {seeds, dist_cal};
}

Neighborhood ti_neighborhood(SortedDataset D_sorted, Point p, double eps, double order)
{
    auto find_result = ranges::find_if(
        D_sorted.data, [&p](const SortedDataset::Row &r) { return r.point.id == p.id; });
    if (find_result == D_sorted.data.end())
        throw std::logic_error("wrong id!");
    auto p_value = find_result->point;
    auto p_index = find_result->point.id;
    auto p_dist = D_sorted.data[p_index].distance_to_reference;
    SortedDataset cut;
    cut.data = D_sorted.data | take(p_index) | to<std::vector>;
    cut.reference = D_sorted.reference;
    auto [backward_neighbors, dist_cal_backward] =
        ti_backward_nieghborhood(cut, D_sorted.data[p_index].point, p_dist, eps, order);
    cut.data = D_sorted.data | drop(p_index) | to<std::vector>;
    auto [forward_neighbors, dist_cal_forward] =
        ti_forward_neighborhood(cut, D_sorted.data[p_index].point, p_dist, eps, order);
    Neighborhood result;
    result.data.insert(
        result.data.end(), backward_neighbors.data.begin(), backward_neighbors.data.end());
    result.data.insert(
        result.data.end(), forward_neighbors.data.begin(), forward_neighbors.data.end());
    result.reference = p_value;
    result.number_of_distance_computations = dist_cal_forward + dist_cal_backward;
    return result;
}

NeighborhoodComputation eps_ti_neighborhood(const Dataset& D, Point reference, double order, double eps)
{
    auto time_now = steady_clock::now();
    auto D_sorted = sort_D(D, reference, order);
    auto time_to_calc_all_ref = steady_clock::now() - time_now;

    auto out_list = std::vector<Neighborhood>{};

    auto nr_of_dist_cal = D.data.size();

    for (const auto& p : D_sorted.data)
    {
        auto neighborhood = ti_neighborhood(D_sorted, p.point, eps, order);
        neighborhood.number_of_distance_computations += nr_of_dist_cal;
        out_list.push_back(std::move(neighborhood));
    }
    return {out_list, duration_cast<milliseconds>(time_to_calc_all_ref)};
}

// void print_return(out_list)
//{
//     for (auto val : out_list)
//     {
//         fmt::print("ID: {}", val[0]);
//         if (val[1] == [])
//         {
//             fmt::print("\tEMPTY");
//             continue;
//         }
//         for (auto ans : val[1])
//         {
//             fmt::print(f "\t{}", ans);
//         }
//     }
// }

void prepare_alg_out(NeighborhoodComputation out_list)
{
    std::vector<Neighborhood> alg_out;
    for (auto val : out_list.neighborhoods)
    {
        alg_out.push_back(val.reference,
                          val.data,
                          val.number_of_distance_computations,
                          val.data.size(),
                          val.data |
                              transform([](DatasetWithReference::Row x) { return x.point.id; }));
    }
    return alg_out;
}

struct AlgParameters
{
    Point r;
    int order;
    double eps;
};

void print_to_file(std::string algorithm_type,
                   std::map<std::string_view, std::string> file_parameters,
                   AlgParameters alg_paremteters,
                   alg_out,
                   std::map<std::string_view, milliseconds> time_out)
{
    auto out_file_path = fmt::format("out_data/{}/OUT_{}_{}_D{}_R{}_m{}_Eps{}.csv",
                                     file_parameters["fname"],
                                     algorithm_type,
                                     file_parameters["fname"],
                                     file_parameters["dimensions"],
                                     file_parameters["rows"],
                                     alg_paremteters.order,
                                     alg_paremteters.eps);

    auto stat_file_path = fmt::format("out_data/{}/STAT_{}_{}_D{}_R{}_m{}_Eps{}.csv",
                                      file_parameters["fname"],
                                      algorithm_type,
                                      file_parameters["fname"],
                                      file_parameters["dimensions"],
                                      file_parameters["rows"],
                                      alg_paremteters.order,
                                      alg_paremteters.eps);
    std::fstream file{out_file_path};
    file << fmt::format("point_id,{},#_of_distance_calculations,|N_Eps|,ids_of_points_in_N_Eps\n",
                        file_parameters["dimensions"]);
    for (auto val : alg_out)
    {
        file << fmt::format("{}", val);
    }
    std::fstream stat_file{stat_file_path};
    stat_file << fmt::format(
        "name_of_the_input_file,#_of_dimensions_of_a_point,#_of_points_in_the_input_file,{},{},"
        "value_of_parameter_Eps,value_of_parameter_m,the_number_of_used_reference_vectors,"
        "reference_vector_1,reading_the_input_file,determining_min_and_max_values_for_each_"
        "dimension,time_calculating_distances_from_each_point_in_the_input_file_to_all_reference_"
        "vectors,total_time,#_of_distance_calculations_between_points_in_the_input_file_and_"
        "reference_vectors,least_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_"
        "of_a_point,greatest_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_"
        "point,avg_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,"
        "variance_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,least_"
        "cardinalities_of_determined_Eps-neighbourhoods,greatest_cardinalities_of_determined_Eps-"
        "neighbourhoods,avg_cardinalities_of_determined_Eps-neighbourhoods,variance_cardinalities_"
        "of_determined_Eps-neighbourhoods",
        file_parameters["dimensions"],
        file_parameters["dimensions"]);

    auto dist_lst = [val[3] for val in alg_out];
    auto card_lst = [val[3] for val in alg_out];


    stat_file << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                                      file_parameters["fname"],
                                      file_parameters["dimensions"],
                                      file_parameters["rows"],
                                      file_parameters["min_values_for_all_dimensions"],
                                      file_parameters["max_values_for_all_dimensions"],
                                      alg_paremteters.eps,
                                      alg_paremteters.order,
                                      algorithm_type == "EPS-NB" ? 0 : 1,
                                      alg_paremteters.r,
                                      time_out["reading_the_input_file"],
                                      time_out["determining_min_and_max_values_for_each_dimension"],
                                      time_out["time_calculating_distances_from_each_point_in_the_input_file_to_all_reference_vectors"],
                                      time_out["total_time"],
                                      accumulate(dist_lst),
                                      min(dist_lst),
                                      max(dist_lst),
                                      average(dist_lst),
                                      var(dist_lst), //variance
                                      min(card_lst),
                                      max(card_lst),
                                      average(card_lst),
                                      var(card_lst) // variance
                             );
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

struct CSVInput
{
    std::string header;
    Dataset dataset;
};

CSVInput datafile_and_transform(std::filesystem::path filename)
{
    std::fstream file{filename};
    std::string line;
    Dataset d;
    std::string header;
    std::getline(file, header);
    while (file.good())
    {
        std::getline(file, line);
        auto tokenized =
            split(line, ",") |
            transform([](const auto &rng)
                      { return std::string_view(&*rng.begin(), ranges::distance(rng)); });
        Point p;
        p.values = tokenized | drop(1) |
                   transform(
                       [](std::string_view s)
                       {
                           double dbl;
                           auto result = std::from_chars(s.data(), s.data() + s.size(), dbl);
                           return dbl;
                       }) |
                   to<std::vector>;
        p.id = (tokenized | take(1) |
                transform(
                    [](std::string_view s)
                    {
                        int dbl;
                        auto result = std::from_chars(s.data(), s.data() + s.size(), dbl);
                        return dbl;
                    }) |
                to<std::vector>)[0];
        d.data.push_back(p);
    };
    return {header, d};
}

struct MinMax
{
    Point min_values;
    Point max_values;
};

MinMax determine_min_and_max_values_for_each_dimension(Dataset D)
{
    auto min_values = D.data[0];
    auto max_values = D.data[0];
    for (auto val : D.data | drop(1))
    {
        for (auto [id_vv, vv] : ranges::view::enumerate(val.values))
        {
            if (min_values.values[id_vv] > vv)
            {
                min_values.values[id_vv] = vv;
            }
            if (max_values.values[id_vv] < vv)
            {
                max_values.values[id_vv] = vv;
            }
        }
    }
    return {.min_values = min_values, .max_values = max_values};
}

int main()
{
    auto input_fname = std::string{"toy_dataset"};
    auto input_filepath = fmt::format("input_data/neighbors/{}.csv", input_fname);
    auto time_now = steady_clock::now();
    auto [file_column_name, D] = datafile_and_transform(input_filepath);
    auto reading_the_input_file = duration_cast<milliseconds>(steady_clock::now() - time_now);
    time_now = steady_clock::now();
    auto [min_values_for_all_dimensions, max_values_for_all_dimensions] =
        determine_min_and_max_values_for_each_dimension(D);
    auto determining_min_and_max_values_for_each_dimension =
        duration_cast<milliseconds>(steady_clock::now() - time_now);

    auto file_parameters = std::map<std::string_view, std::string>{
        {"fname", input_fname},
        {"dimensions", file_column_name[1]},
        {"rows", D.data.size()},
        {"min_values_for_all_dimensions", min_values_for_all_dimensions},
        {"max_values_for_all_dimensions", max_values_for_all_dimensions}};

    auto alg_paremteters = AlgParameters{.r = {.values{0.0}}, .order = 2, .eps = 1.5};

    auto r = Point{.values{0.0, 0.0}};
    auto m = 2;
    auto eps = 1.5;

    fmt::print("EPS-NB");

    time_now = steady_clock::now();

    auto [out_list, time_to_calc_all_ref] = eps_neighborhood(D, m, eps = eps);

    auto total_time = duration_cast<milliseconds>(steady_clock::now() - time_now);

    auto alg_out = prepare_alg_out(out_list);

    auto time_out = std::map<std::string_view, milliseconds>{
        {"reading_the_input_file", reading_the_input_file},
        {"determining_min_and_max_values_for_each_dimension",
         determining_min_and_max_values_for_each_dimension},
        {"time_calculating_distances_from_each_point_in_the_input_file_to_all_"
         "reference_vectors",
         time_to_calc_all_ref},
        {"total_time", total_time}};
    print_to_file("EPS-NB", file_parameters, alg_paremteters, alg_out, time_out);

    fmt::print("EPS-TI-NB");

    time_now = steady_clock::now();
    auto [eps_ti_out_list, eps_ti_time_to_calc_all_ref] = eps_ti_neighborhood(D, r, m, eps = eps);
    total_time = duration_cast<milliseconds>(steady_clock::now() - time_now);

    alg_out = prepare_alg_out(out_list);

    time_out = std::map<std::string_view, milliseconds>{
        {"reading_the_input_file", reading_the_input_file},
        {"determining_min_and_max_values_for_each_dimension",
         determining_min_and_max_values_for_each_dimension},
        {"time_calculating_distances_from_each_point_in_the_input_file_"
         "to_all_reference_vectors",
         time_to_calc_all_ref},
        {"total_time", total_time}};

    print_to_file("EPS-TI-NB", file_parameters, alg_paremteters, alg_out, time_out);
}
