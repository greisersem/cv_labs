cmake_minimum_required(VERSION 3.16)
project(lab_5)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

include_directories(${OPENCV4_INCLUDE_DIRS})
link_directories(${OPENCV4_LIBRARY_DIRS})

add_executable(lab_5 lab_5.cpp)

target_link_libraries(lab_5 ${OPENCV4_LIBRARIES})