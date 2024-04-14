
#include "char_metric.hpp"

#include <iostream>

using namespace sharded_map;

// Example using stateful batch insertion.
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

  // Create a map for 4 threads with a queue capacity of 128 elements.
  Map map(NUM_THREADS, 128);

  // This will be our thread-local state. It just counts how many chars a thread has handled.
  struct CharCounter {
    size_t thread_id;
    size_t count;
  };

  // Iterate from 0 (inclusive) to s.size() (exclusive)
  // The first lambda takes the thread id as the argument and returns the initial state for that
  // thread. The state can be of any desired type. The second lambda generates new values to insert
  // into the map. i is the current index in the iteration and the second argument is a reference to
  // the current thread-local state, which in our case is a CharCounter.
  std::vector<CharCounter> char_counters = map.batch_insert(
      0,
      s.size(),
      [](size_t thread_id) { return CharCounter{thread_id, 0}; },
      [&s](size_t i, CharCounter &counter) {
        counter.count++;
        return std::pair(s[i], i);
      });

  for (const CharCounter &cc : char_counters) {
    std::cout << "Thread " << cc.thread_id << " has handled " << cc.count << " chars" << std::endl;
  }

  // Iterate over the map. and execute a function
  map.for_each([](const char &k, const CharMetric &v) {
    std::cout << k << " appears " << v.count << " times and first at position " << v.first_occ
              << std::endl;
  });

  return 0;
}
