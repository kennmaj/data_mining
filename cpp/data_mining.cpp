#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string_regex.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fmt/chrono.h"

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
#include <chrono>
#include <map>
#include <filesystem>

using namespace std::chrono_literals;
using namespace std::chrono;
using namespace ranges;
using namespace ranges::view;
using namespace std::literals;

class Timed
{
public:
    template<typename ReturnType> struct Result
    {
        ReturnType result;
        std::chrono::nanoseconds execution_time;
    };
    template<typename ReturnType, std::invocable T, typename... Args>
    ReturnType operator()(T t, Args... args)
    {
        auto before = std::chrono::steady_clock::now();
        Result<ReturnType> result;
        result.result = t(std::forward<Args>(args)...);
        auto after = std::chrono::steady_clock::now();
        result.execution_time = after - before;
        return result;
    }
};

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

SortedDataset sortDataset(Dataset D, const Point &reference_point, double order)
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
    std::size_t distance_to_reference_calls;
    std::chrono::nanoseconds computation_time;
};

NeighborhoodComputation epsilonNeighborhood(Dataset D, int minkowski_distance_order, double eps)
{
    using namespace std::chrono;
    auto ids = transform(D.data, [](const Point &x) { return x.id; });
    auto out_list = std::vector<Neighborhood>{};
    auto time_to_calc_all_ref = 0ns;
    for (auto p : ids)
    {
        auto find_result = find_if(D.data, [&p](const Point &r) { return r.id == p; });
        if (find_result == D.data.end())
            throw std::logic_error("wrong id!");
        auto p_value = *find_result;
        auto time_now = steady_clock::now();
        auto D_copy = D;
        std::erase_if(D_copy.data, [&p_value](const Point &r) { return r.id == p_value.id; });
        auto D_sorted = sortDataset(D_copy, p_value, minkowski_distance_order);
        time_to_calc_all_ref += steady_clock::now() - time_now;
        Neighborhood n;
        n.reference = p_value;
        n.data = filter(D_sorted.data, [&eps](auto x) { return x.distance_to_reference <= eps; }) |
                 to<std::vector>;
        n.number_of_distance_computations = D_sorted.data.size();
        out_list.emplace_back(n);
    }
    return {.neighborhoods = out_list, .distance_to_reference_calls=0,.computation_time = time_to_calc_all_ref};
}

Neighborhood TIBackwardNeighborhood(
    SortedDataset D, const Point &p, double p_dist, double eps, double order)
{
    auto seeds = Neighborhood{};
    seeds.number_of_distance_computations = 0;
    auto backward_threshold = p_dist - eps;
    for (auto i = D.data.size() - 1; i != -1; --i)
    {
        auto q = D.data[i];
        if (q.distance_to_reference < backward_threshold)
        {
            break;
        }
        auto dist_val = q.point.minkowskiDistanceTo(p, order);
        seeds.number_of_distance_computations++;
        if (dist_val <= eps)
        {
            Neighborhood::Row r;
            r.point = q.point;
            r.distance_to_reference = dist_val;
            seeds.data.push_back(r);
        }
    }
    return seeds;
}

Neighborhood TIForwardNeighborhood(
    const DatasetWithReference &D, Point p, double p_dist, double eps, double order)
{
    auto seeds = Neighborhood{};
    seeds.number_of_distance_computations = 0;
    auto forward_threshold = eps + p_dist;
    for (const auto &q : D.data)
    {
        if (q.distance_to_reference > forward_threshold)
        {
            break;
        }
        auto dist_val = q.point.minkowskiDistanceTo(p, order);
        seeds.number_of_distance_computations++;
        if (dist_val <= eps)
        {
            auto r = Neighborhood::Row{};
            r.point = q.point;
            r.distance_to_reference = dist_val;
            seeds.data.push_back(r);
        }
    }
    return seeds;
}

Neighborhood TINeighborhood(SortedDataset D_sorted, Point p, double eps, double order)
{
    auto find_result = ranges::find_if(
        D_sorted.data, [&p](const SortedDataset::Row &r) { return r.point.id == p.id; });
    if (find_result == D_sorted.data.end())
        throw std::logic_error("wrong id!");
    auto p_value = find_result->point;
    auto p_index = std::distance(D_sorted.data.begin(), find_result);
    auto p_dist = find_result->distance_to_reference;
    SortedDataset cut;
    cut.data = D_sorted.data | take(p_index) | to<std::vector>;
    cut.reference = D_sorted.reference;
    auto backward_neighbors =
        TIBackwardNeighborhood(cut, D_sorted.data[p_index].point, p_dist, eps, order);
    cut.data = D_sorted.data | drop(p_index + 1) | to<std::vector>;
    auto forward_neighbors =
        TIForwardNeighborhood(cut, D_sorted.data[p_index].point, p_dist, eps, order);
    Neighborhood result;
    result.data.insert(
        result.data.end(), backward_neighbors.data.begin(), backward_neighbors.data.end());
    result.data.insert(
        result.data.end(), forward_neighbors.data.begin(), forward_neighbors.data.end());
    result.reference = p_value;
    result.number_of_distance_computations = backward_neighbors.number_of_distance_computations +
                                             forward_neighbors.number_of_distance_computations;
    return result;
}

NeighborhoodComputation epsilonTINeighborhood(const Dataset &D,
                                            Point reference,
                                            double order,
                                            double eps)
{
    auto time_now = steady_clock::now();
    auto D_sorted = sortDataset(D, reference, order);
    auto time_to_calc_all_ref = steady_clock::now() - time_now;

    auto out_list = std::vector<Neighborhood>{};

    for (const auto &p : D_sorted.data)
    {
        auto neighborhood = TINeighborhood(D_sorted, p.point, eps, order);
        out_list.push_back(std::move(neighborhood));
    }
    return {.neighborhoods=out_list,.distance_to_reference_calls=D.data.size(),.computation_time= time_to_calc_all_ref};
}

struct AlgParameters
{
    Point r;
    int order;
    double eps;
    std::string rval;
};

void printToFile(std::string algorithm_type,
                   std::map<std::string_view, std::string> file_parameters,
                   AlgParameters alg_parameters,
                   NeighborhoodComputation alg_out,
                   std::map<std::string_view, nanoseconds> time_out)
{
    using std::filesystem::path;
    auto common_suffix =  fmt::format("{}_{}_D{}_R{}_m{}_Eps{:.2f}_r{}-cpp.csv",
                                     algorithm_type,
                                     file_parameters["fname"],
                                     file_parameters["number_of_dimensions"],
                                     file_parameters["rows"],
                                     alg_parameters.order,
                                     alg_parameters.eps,
                                     alg_parameters.rval);
    auto out_file_path = path("out_data") /file_parameters["fname"] / path("OUT_" + common_suffix);
    auto stat_file_path = path("out_data") /file_parameters["fname"] / path("STAT_" + common_suffix);
    std::ofstream file{out_file_path};
    std::vector<std::string> tokenized;
    boost::algorithm::split(tokenized, file_parameters["dimensions"], boost::is_any_of(","));
    auto header = std::string{};
    for (auto token : tokenized | drop(1))
    {
        header += fmt::format("v{},", token);
    }
    file << fmt::format("point_id,{}#_of_distance_calculations,|N_Eps|,ids_of_points_in_N_Eps\n",header);
    for (auto val : alg_out.neighborhoods)
    {
        file << fmt::format("{},{},{},{},[{}]\n",
                            val.reference.id,
                            fmt::join(val.reference.values, ","),
                            val.number_of_distance_computations,
                            val.data.size(),
                            fmt::join(val.data | transform([](auto x) { return x.point.id; })," "));
    }
    std::ofstream stat_file{stat_file_path};

    auto min_header = std::string{};
    for (auto token : tokenized | drop(1))
    {
        min_header += fmt::format("min{},", token);
    }
    auto max_header = std::string{};
    for (auto token : tokenized | drop(1))
    {
        max_header += fmt::format("max{},", token);
    }
    // clang-format off
    stat_file << fmt::format(
        "name_of_the_input_file,"
        "#_of_dimensions_of_a_point,"
        "#_of_points_in_the_input_file,"
        "{}"
        "{}"
        "value_of_parameter_Eps,"
        "value_of_parameter_m,"
        "the_number_of_used_reference_vectors,"
        "reference_vector_1,"
        "reading_the_input_file,"
        "determining_min_and_max_values_for_each_dimension,"
        "time_calculating_distances_from_each_point_in_the_input_file_to_all_reference_vectors,"
        "total_time,"
        "#_of_distance_calculations_between_points_in_the_input_file_and_reference_vectors,"
        "least_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,"
        "greatest_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,"
        "avg_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,"
        "variance_#_of_distance_calculations_carried_out_to_find_Eps-neigbourhood_of_a_point,"
        "least_cardinalities_of_determined_Eps-neighbourhoods,"
        "greatest_cardinalities_of_determined_Eps-neighbourhoods,"
        "avg_cardinalities_of_determined_Eps-neighbourhoods,"
        "variance_cardinalities_of_determined_Eps-neighbourhoods\n",
        min_header,
        max_header);
    // clang-format on
    auto dist_lst = alg_out.neighborhoods | transform([](const Neighborhood &x)
                                        { return x.number_of_distance_computations; });
    auto card_lst = alg_out.neighborhoods | transform([](const Neighborhood &x) { return x.data.size(); });

    double avg_dist = accumulate(dist_lst, 0.0) / dist_lst.size();
    double var_dist = (accumulate(dist_lst,
                                  0.0,
                                  [avg_dist](auto sum, auto elem)
                                  { return sum + (elem - avg_dist) * (elem - avg_dist); }) /
                       dist_lst.size());
    double avg_card = accumulate(card_lst, 0.0) / card_lst.size();
    double var_card = (accumulate(card_lst,
                                  0.0,
                                  [avg_card](auto sum, auto elem)
                                  { return sum + (elem - avg_card) * (elem - avg_card); }) /
                       dist_lst.size());
    stat_file << fmt::format(
        "{},{},{},{},{},{},{},{},[{}],{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
        file_parameters["fname"],
        file_parameters["number_of_dimensions"],
        file_parameters["rows"],
        file_parameters["min_values_for_all_dimensions"],
        file_parameters["max_values_for_all_dimensions"],
        alg_parameters.eps,
        alg_parameters.order,
        algorithm_type == "EPS-NB" ? 0 : 1,
        algorithm_type == "EPS-NB" ? "N/A" : fmt::format("{}",fmt::join(alg_parameters.r.values, " ")),
        time_out["reading_the_input_file"],
        time_out["determining_min_and_max_values_for_each_dimension"],
        time_out["time_calculating_distances_from_each_point_in_the_input_"
                 "file_to_all_reference_vectors"],
        time_out["total_time"],
        alg_out.distance_to_reference_calls,
        min(dist_lst),
        max(dist_lst),
        avg_dist,
        var_dist, // variance
        min(card_lst),
        max(card_lst),
        avg_card,
        var_card // variance
    );
}

struct CSVInput
{
    std::string header;
    Dataset dataset;
};

CSVInput loadDataset(const std::filesystem::path &filename)
{
    std::fstream file{filename};
    Dataset d;
    std::string tokenized_header;
    std::getline(file, tokenized_header);
    while (file.good())
    {
        std::string line;
        std::getline(file, line);
        if (line.empty())
        {
            break;
        }
        std::vector<std::string> tokenized;
        boost::algorithm::split(tokenized, line, boost::is_any_of(","));
        //        boost::algorithm::split_regex(tokenized, line, boost::regex( ", " ) ) ;
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
    return {tokenized_header, d};
}

struct MinMax
{
    Point min_values;
    Point max_values;
};

MinMax calculateVectorsWithExtremeValues(Dataset D)
{
    auto min_values = D.data[0];
    auto max_values = D.data[0];
    for (auto vector : D.data | drop(1))
    {
        for (auto [index, value] : ranges::view::enumerate(vector.values))
        {
            if (min_values.values[index] > value)
            {
                min_values.values[index] = value;
            }
            if (max_values.values[index] < value)
            {
                max_values.values[index] = value;
            }
        }
    }
    return {.min_values = min_values, .max_values = max_values};
}

template<class... Ts> struct Overload : Ts...
{
    using Ts::operator()...;
};

void runTestCase(std::string input_fname,
                   std::string rval,
                   int m,
                   std::variant<double, std::string> eps)
{
    auto input_filepath = fmt::format("input_data/data/{}.csv", input_fname);

    auto time_now = steady_clock::now();
    auto [file_column_name, dataset] = loadDataset(input_filepath);
    auto reading_the_input_file = steady_clock::now() - time_now;
    time_now = steady_clock::now();
    auto [min_vector, max_vector] =
        calculateVectorsWithExtremeValues(dataset);
    auto time_to_calculate_extremes = steady_clock::now() - time_now;
    auto file_parameters = std::map<std::string_view, std::string>{
        {"fname", input_fname},
        {"number_of_dimensions", fmt::format("{}", min_vector.values.size())},
        {"dimensions", file_column_name},
        {"rows", fmt::format("{}", dataset.data.size())},
        {"min_values_for_all_dimensions",
         fmt::format("{}", fmt::join(min_vector.values, ","))},
        {"max_values_for_all_dimensions",
         fmt::format("{}", fmt::join(max_vector.values, ","))}};

    Point r;
    if (rval == "max")
    {
        r = max_vector;
    }
    else if (rval == "min")
    {
        r = min_vector;
    }
    else if (rval == "0")
    {
        r.values.resize(max_vector.values.size(), 0.0);
    }

    double epsval = std::visit(
        Overload{[&max_vector, &min_vector](std::string s)
                 {
                     double pom = std::stoi(s);
                     auto result = max_vector.minkowskiDistanceTo(min_vector, 2) /
                            pom;
                        char buffer[100];
                     snprintf(buffer, sizeof(buffer), "%.1f", result);
                     return std::stod(buffer);
                 },
                 [](double s) { return s; }},
        eps);
    fmt::println("rval: {}, m: {}, Eps: {}", rval, m, epsval);

    auto alg_paremteters = AlgParameters{.r = r, .order = m, .eps = epsval, .rval = rval};

    fmt::println("EPS-TI-NB");

    time_now = steady_clock::now();
    auto result = epsilonTINeighborhood(dataset, r, m, epsval);
    auto total_time = steady_clock::now() - time_now;

    // alg_out = prepare_alg_out(out_list = out_list)
    auto time_out = std::map<std::string_view, nanoseconds>{
        {"reading_the_input_file", reading_the_input_file},
        {"determining_min_and_max_values_for_each_dimension", time_to_calculate_extremes},
        {"time_calculating_distances_from_each_point_in_the_input_file_"
         "to_all_reference_vectors",
         result.computation_time},
        {"total_time", total_time}};

    printToFile("EPS-TI-NB", file_parameters, alg_paremteters, result, time_out);

    fmt::println("EPS-NB");

    time_now = steady_clock::now();

    result = epsilonNeighborhood(dataset, m, epsval);

    total_time = steady_clock::now() - time_now;

    time_out = std::map<std::string_view, nanoseconds>{
        {"reading_the_input_file", reading_the_input_file},
        {"determining_min_and_max_values_for_each_dimension", time_to_calculate_extremes},
        {"time_calculating_distances_from_each_point_in_the_input_file_"
         "to_all_reference_vectors",
         result.computation_time},
        {"total_time", total_time}};
    file_parameters = std::map<std::string_view, std::string>{
        {"fname", input_fname},
        {"number_of_dimensions", fmt::format("{}", min_vector.values.size())},
        {"dimensions", file_column_name},
        {"rows", fmt::format("{}", dataset.data.size())},
        {"min_values_for_all_dimensions",
         fmt::format("{}", fmt::join(min_vector.values, ","))},
        {"max_values_for_all_dimensions",
         fmt::format("{}", fmt::join(max_vector.values, ","))}};
    printToFile("EPS-NB", file_parameters, alg_paremteters, result, time_out);
}

int main()
{
    auto input_fname = std::string{"toy_dataset"};
    //    auto input_fname = "wine_quality";
    //    auto  input_fname = "2d_elastodynamic_metamaterials";a
    //    input_fname = "dry_bean_dataset" # 35

    auto r = std::string{"0"};
    auto m = 2;
    auto eps = std::variant<double, std::string>{1.8};
//    runTestCase(input_fname, r, m, eps);

        for (auto input_fname : {"toy_dataset"s, "2d_elastodynamic_metamaterials"s,
        "dry_bean_dataset"s, "wine_quality"s })
        {
            fmt::println("{}",input_fname);
            m = 2;
            eps = "50";
            r = "0";
            runTestCase(input_fname, r, m, eps);
            r = "max";
            runTestCase(input_fname, r, m, eps);
            r = "min";
            runTestCase(input_fname, r, m, eps);
            r = "0";
            eps = "45";
            runTestCase(input_fname, r, m, eps);
            eps = "55";
            runTestCase(input_fname, r, m, eps);
            eps = "50";
            m = 1;
            runTestCase(input_fname, r, m, eps);
            m = 3;
            runTestCase(input_fname, r, m, eps);
        }
}
