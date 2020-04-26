cmake_minimum_required (VERSION 3.12)
project (tensorflow_example)

set (CMAKE_CXX_STANDARD 17)
if(APPLE)
  # On MacOS:
  #   -undefined dynamic_lookup is necessary for pybind11 linking
  set (CMAKE_CXX_FLAGS "-Wno-everything -w -undefined dynamic_lookup")

  # On MacOS, we need this so that CMake will use the right Python if the user
  # has a virtual environment active
  set (CMAKE_FIND_FRAMEWORK LAST)
else()
  set (CMAKE_CXX_FLAGS "-Wno-everything")
endif()

# Position-independent code is needed for Python extension modules.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# We add the subdirectory here so open_spiel_core can #include absl.
add_subdirectory (abseil-cpp)
include_directories (abseil-cpp)
link_libraries (
  absl::container
  absl::flags
  absl::flags_parse
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

add_executable(loading_example loading_example.cc)
target_include_directories(loading_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(loading_example tensorflow)

add_executable(training_example training_example.cc)
target_include_directories(training_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(training_example tensorflow)