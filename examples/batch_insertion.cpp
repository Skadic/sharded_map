#include "char_metric.hpp"

#include <iostream>

using namespace sharded_map;

// Example using batch insertion.
// For more extensive descriptions, look at the manual_insertion example
int main() {
  // Seed random number generator
  std::srand(std::time(nullptr));

  // Fill a string with 100000 random chars from 'a' to 'z'
  std::string s;
  s.resize(100'000);
  std::generate(s.begin(), s.end(), []() { return std::rand() % 26 + 'a'; });

  constexpr size_t NUM_THREADS = 4;

  // Type definition for our map
  using Map = ShardedMap<char, CharMetric, std::unordered_map, FindChar>;

  // Create a map for 4 threads with a queue capacity if 128 elements.
  Map map(NUM_THREADS, 128);

  // Iterate from 0 (inclusive) to s.size() (exclusive)
  // The i in the lambda is the current index in the iteration. 
  // One invocation of the lambda creates a single key-value pair that is inserted into the map.
  map.batch_insert(0, s.size(), [&s](size_t i) { return std::pair(s[i], i); });

  // Iterate over the map. and execute a function
  map.for_each([](const char &k, const CharMetric &v) {
    std::cout << k << " appears " << v.count << " times and first at position " << v.first_occ
              << std::endl;
  });

  return 0;
}
