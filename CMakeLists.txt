cmake_minimum_required(VERSION 3.0)

project(selective_search)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

find_package(PythonLibs 3.0 REQUIRED)
if(PYTHONLIBS_FOUND)
  include_directories(${PYTHON_INCLUDE_DIRS})
endif()

option(WITH_SP "test SP" FALSE)

if(WITH_SP)
  message("true")
  find_package(Boost 1.58 COMPONENTS python3 numpy REQUIRED)
  if(Boost_FOUND)
    include_directories(${CMAKE_SOURCE_DIR}/segment ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
    add_library(segment SHARED segment_py.cpp)
    target_link_libraries(segment ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
    set_target_properties(segment PROPERTIES PREFIX "" SUFFIX ".so")
  endif()
else()
  message("false")
endif()
