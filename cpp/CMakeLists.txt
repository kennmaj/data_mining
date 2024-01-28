cmake_minimum_required(VERSION 3.27)
project(ESDAMI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

#include(cmake/dependencies.cmake)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
#add_subdirectory(sources)
#add_subdirectory(applications)

add_executable(data_mining
        data_mining.cpp)
target_link_libraries(data_mining PRIVATE
        Eigen3::Eigen
        fmt::fmt
        range-v3::range-v3
        Boost::regex
        Boost::boost
)
