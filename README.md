# Sharded Map

A sharded hash map implementation for parallel processing written for C++20.

The central type is `ShardedMap<K, V, SeqHashMapType, UpdateFn>`.
- `K` is the type of the keys.
- `V` is the type of the values.
- `SeqHashMapType` is the type of the hash map each thread will have locally.
  Any `std::unordered_map` compatible map should work.
- `UpdateFn` is an *update function* denoting what happens, if a new value is inserted or an existing value is updated.
  For examples, check the `sharded_map::update_functions` namespace or the `examples/` folder in this repository.

Documentation is available [here](https://skadic.github.io/sharded_map).

## Usage

This is a header-only library, so you can just use the `sharded_map.hpp` header as is.
If you use CMake, you can for example add this repo as a git submodule to your project, and add this to your `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/this/repo/sharded_map)

target_link_libraries(your_target PRIVATE sharded_map)
```

## Tests

You can build and run tests using these commands:
```bash
cmake --preset=test
cmake --build --preset=test --config=Release
ctest --preset=default
```

## Example

```cpp
#include <sharded_map/sharded_map.hpp>
#include <atomic>
#include <omp.h>

using namespace sharded_map;

int main() {
  using Map = ShardedMap<int, char, std::unordered_map, update_functions::Overwrite<size_t, size_t>>;

  Map map(4, 128);
  // Iterate in parallel from 0 to 100'000 and insert the pair (i, 2*i) for each index
  map.batch_insert(0, 100'000, [&s](size_t i) { return std::pair(i, 2*i); });
}

```

For a more extensive and commented example, check the `examples/` folder.
You can build the examples using these commands:

```bash
cmake --preset=examples
cmake --build --preset=examples --config=Release
```

Then you can find the built binaries in `./build/examples/Release/`.
