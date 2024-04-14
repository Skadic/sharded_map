include(ExternalProject)
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(sharded_map_test sharded_map_test.cpp)
target_link_libraries(sharded_map_test PRIVATE Catch2::Catch2WithMain sharded_map)
target_include_directories(sharded_map_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(sharded_map_test)
