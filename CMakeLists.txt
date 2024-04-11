cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(sharded_map)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

# Build tests
option(SHARDED_MAP_BUILD_TESTS "Build sharded hash map tests." OFF)

add_library(sharded_map INTERFACE)
target_include_directories(sharded_map
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(SHARDED_MAP_BUILD_TESTS)
  add_subdirectory(test)
endif()
