cmake_minimum_required(VERSION 3.12)

project(VO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -O2")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB HEADERS "*.h")

add_executable(VO main.cpp vo.cpp ${HEADERS})

target_link_libraries(VO ${OpenCV_LIBS})