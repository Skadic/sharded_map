# Sharded Map

A sharded hash map implementation for parallel processing written for C++20.

The central type is `ShardedMap<K, V, SeqHashMapType, UpdateFn>`.
- `K` is the type of the keys.
- `V` is the type of the values.
- `SeqHashMapType` is the type of the hash map each thread will have locally.
  Any `std::unordered_map` compatible map should work.
- `UpdateFn` is an *update function* denoting what happens, if a new value is inserted or an existing value is updated.
  For examples, check the `sharded_map::update_functions` namespace or the `examples/` folder in this repository.

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
  using Map = ShardedMap<int, char, std::unordered_map, update_functions::Overwrite<int, char>>;

  Map                map(4, 128);
  std::atomic_size_t threads_done;

#pragma omp parallel num_threads(4)
  {
    const size_t thread_id = omp_get_thread_num();
    const size_t start     = thread_id * 250'000;
    const size_t end       = (thread_id + 1) * 250'000;
    Map::Shard   shard     = map.get_shard(thread_id);

    auto &barrier = map.barrier();

    for (size_t i = start; i < end; i++) {
      shard.insert(i, i * 2);
    }

    threads_done++;

    while (threads_done.load() < 4) {
      shard.handle_queue_sync(false);
    }
    barrier.arrive_and_drop();
  }
}

```

For a more extensive and commented example, check the `examples/` folder.
You can build the examples using these commands:

```bash
cmake --preset=examples
cmake --build --preset=examples --config=Release
```

Then you can find the built binaries in `./build/examples/Release/`.
