cmake_minimum_required(VERSION 3.16)
project(supervoxel)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "-lboost_program_options")
set(CMAKE_CXX_FLAGS "-lcnpy -lz")
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
find_package(PCL REQUIRED)

include_directories(
${Boost_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)
MESSAGE("${Boost_LIBRARIES}")
add_executable(supervoxel main.cpp semkitti.cpp s3dis.cpp scannet.cpp)
target_link_libraries(supervoxel ${PCL_LIBRARIES} ${Boost_LIBRARIES} /usr/local/lib/libcnpy.so libboost_program_options.a)
