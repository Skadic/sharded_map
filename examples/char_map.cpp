
#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>

#include <sharded_map/sharded_map.hpp>

/// Tracks when a char first appears and how often it appears.
struct CharMetric {
  /// The character
  char c;
  /// The first occurrence
  size_t first_occ;
  /// The char's number of occurrences
  size_t count;
};

/// This is an update function that counts how often a character appears and the first text position
/// at which it appears. It is an update function for a Sharded Map with keys of type char and
/// values of type CharMetric
///
/// Check out the sharded_map::update_functions namespace for some predefined basic update
/// functions.
struct FindChar {
  /// This is the type that is the parameter to insert calls.
  /// When we report a character, we want its text position, which is a size_t.
  using InputValue = size_t;

  /// This function is called when a char already exists in the map.
  ///
  /// The arguments are the key, a reference to the associated value saved in the map, and the value
  /// that was just inserted. You can modify the reference to the value to your heart's content.
  ///
  /// In our case, the key is a char, the associated value is the char metric and the inserted value
  /// is the text position. Here we don't need the key, so we just ignore it.
  static void update(const char &, CharMetric &metric, InputValue &&position) {
    // Increment number of occurrences of char
    metric.count++;
    // Update the first position of that char
    metric.first_occ = std::min(metric.first_occ, position);
  };

  /// This function is called when the map did not previously contain the key.
  /// It should return the initial value of the key's associated value after insertion.
  ///
  /// The arguments are the key and the inserted value.
  ///
  /// In our case, if a character appears for the first time, it of course appeared once so far.
  static CharMetric init(const char &key, InputValue &&position) {
    return CharMetric{key, position, 1};
  }
};

using namespace sharded_map;

// Check that FindChar actually fulfills the "UpdateFunction" concept
// for a sharded map with keys of type char and values of type CharMetric
static_assert(UpdateFunction<FindChar, char, CharMetric>,
              "FindChar does not fulfill UpdateFunction concept");

int main() {
  // Seed random number generator
  std::srand(std::time(nullptr));

  // Fill a string with 100000 random chars from 'a' to 'z'
  std::string s;
  s.resize(100'000);
  std::generate(s.begin(), s.end(), []() { return std::rand() % 26 + 'a'; });

  constexpr size_t NUM_THREADS = 4;

  // Type definition for our map
  // This is a sharded hash map whose keys are chars and whose values are CharMetric.
  // Each thread has an std::unordered_map as a local hash map. You can use any compatible hash map
  // implementation instead. I recommend ankerl::unordered_dense::map from
  // https://github.com/martinus/unordered_dense
  //
  // We use the FindChar update function to decide what happens when inserting or updating values.
  // Since the keys are chars and CountChar::InputType = size_t,
  // we insert pairs of std::pair<char, size_t> when calling insert.
  using Map = ShardedMap<char, CharMetric, std::unordered_map, FindChar>;

  // Create a map for 4 threads with a queue capacity if 128 elements.
  Map map(NUM_THREADS, 128);

  // We keep track of how many threads are done inserting values.
  // This allows threads to know when they can stop handling their queues.
  std::atomic_size_t threads_done;

  // Parallel area
#pragma omp parallel num_threads(NUM_THREADS)
  {
    // Create a shard for each thread. We access the map using this shard. The argument is the
    // thread id. E.g. for 4 threads, the thread ids *must* be 0,1,2,3 for each thread respectively.
    const size_t thread_id = omp_get_thread_num();
    const size_t start     = thread_id * (s.size() / NUM_THREADS);
    const size_t end       = (thread_id + 1) * (s.size() / NUM_THREADS);
    Map::Shard   shard     = map.get_shard(thread_id);

    // The thread barrier used by the sharded map
    // This contains the number of threads required in `handle_queue_sync`
    // for all threads to unblock and handle the queues.
    auto &barrier = map.barrier();

    // Insert the chars and text positions.
    for (size_t i = start; i < end; i++) {
      shard.insert(s[i], i);
    }

    // This thread is done inserting
    threads_done++;

    // Threads might still be stuck in the above loop, filling up other thread's queues
    // The threads who finished early will wait until queues need emptying,
    // but will not force other threads to empty their queues when they try to insert.
    // (this last thing is what the `false` parameter does)
    while (threads_done.load() < NUM_THREADS) {
      shard.handle_queue_sync(false);
    }
    // This thread is done participating in the map. If any threads are still stuck in the above
    // loop, they won't be waiting for this thread anymore. I.e. this decrements the number of
    // threads required to unblock the barrier.
    barrier.arrive_and_drop();
  }

  // IMPORTANT: if you wish to insert into the map after this point, you must call this:
  // As we modified the barrier by using arrive_and_drop, we need to reset it to its former state.
  // You don't need to do this if you don't plan to insert into the map after this point.
  map.reset_barrier();

  // Iterate over the map. and execute a function
  map.for_each([](const char &k, const CharMetric &v) {
    std::cout << k << " appears " << v.count << " times and first at position " << v.first_occ
              << std::endl;
  });

  // map.find() works as expected
  auto result = map.find('a');
  if (result != map.end()) {
    const auto &[key, value] = *result;
    std::cout << "result for char '" << key << "' found! " << value.count << " " << value.first_occ
              << std::endl;
  }

  a();

  return 0;
}
