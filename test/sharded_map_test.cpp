#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <omp.h>
#include <ranges>
#include <string>
#include <unordered_map>

#include <char_metric.hpp>
#include <sharded_map/sharded_map.hpp>

constexpr size_t NUM_ELEMS   = 100'000;
constexpr size_t NUM_THREADS = 4;

using namespace sharded_map;

TEST_CASE("basic insertions", "[ints]") {
  using Map =
      ShardedMap<size_t, size_t, std::unordered_map, update_functions::Overwrite<size_t, size_t>>;
  Map map(NUM_THREADS, 128);

  std::vector<size_t> values(NUM_ELEMS);
  std::iota(values.begin(), values.end(), 0);
  std::atomic_size_t threads_done;

  // Insert tuples (i,i) into the map
#pragma omp parallel num_threads(NUM_THREADS)
  {
    // Create a shard for each thread
    Map::Shard shard   = map.get_shard(omp_get_thread_num());
    auto      &barrier = map.barrier();

    // Insert all values
    for (size_t i = 0; i < values.size(); i++) {
      shard.insert(i, values[i]);
    }

    // Mark that another thread is done
    threads_done++;

    // We wait for the other threads to be done.
    // As queues might still fill up, each thread handles its queue
    // However, we won't force the other threads to handle their queues
    while (threads_done.load() < NUM_THREADS) {
      shard.handle_queue_sync(false);
    }
    // Empty the rest of the queue. All threads are done with the insert loop, so there won't be any
    // more insertions.
    shard.handle_queue_async();
    // This thread is completely done and we won't make the other threads wait for this one,
    // if they are still stuck in the loop above
    barrier.arrive_and_drop();
  }

  SECTION("size equal to the number of values after insert") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after insert") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are inserted") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }

  // calling arrive_and_drop modifies the barrier permanently. see the docs for std::barrier on
  // cppreference
  map.reset_barrier();
  threads_done = 0;

  for (size_t &v : values) {
    v *= 2;
  }

  // We insert the tuples (i, 2*i). Since we used the Overwrite Update function, this should
  // overwrite the values
#pragma omp parallel num_threads(NUM_THREADS)
  {
    Map::Shard shard   = map.get_shard(omp_get_thread_num());
    auto      &barrier = map.barrier();

    for (size_t i = 0; i < values.size(); i++) {
      shard.insert(i, 2 * i);
    }

    threads_done++;

    while (threads_done.load() < NUM_THREADS) {
      shard.handle_queue_sync(false);
    }
    shard.handle_queue_async();
    barrier.arrive_and_drop();
  }

  SECTION("size equal to the number of values after updates") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after updates") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are updated") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }
}

TEST_CASE("batch insertions", "[batch] [ints]") {
  using Map =
      ShardedMap<size_t, size_t, std::unordered_map, update_functions::Overwrite<size_t, size_t>>;
  Map map(NUM_THREADS, 128);

  std::vector<size_t> values(NUM_ELEMS);
  std::iota(values.begin(), values.end(), 0);

  map.batch_insert((size_t) 0, NUM_ELEMS, [&values](size_t i) { return std::pair(i, values[i]); });

  SECTION("size equal to the number of values after insert") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after insert") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are inserted") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }

  for (size_t &v : values) {
    v *= 2;
  }

  map.batch_insert((size_t) 0, NUM_ELEMS, [&values](size_t i) { return std::pair(i, values[i]); });

  SECTION("size equal to the number of values after updates") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after updates") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are updated") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }
}

TEST_CASE("batch state insertions", "[batch] [state] [ints]") {
  using Map =
      ShardedMap<size_t, size_t, std::unordered_map, update_functions::Overwrite<size_t, size_t>>;
  Map map(NUM_THREADS, 128);

  constexpr size_t SEGMENT_SIZE = util::ceil_div(NUM_ELEMS, NUM_THREADS);

  std::vector<size_t> values(NUM_ELEMS);
  {
    size_t i = 0;
    std::generate(values.begin(), values.end(), [&i]() { return i++ % SEGMENT_SIZE; });
  }

  map.batch_insert((size_t) 0,
                   NUM_ELEMS,
                   [](size_t) -> size_t { return 0; },
                   [](size_t i, size_t &state) {
                     const size_t old = state;
                     state++;
                     return std::pair(i, old);
                   });

  SECTION("size equal to the number of values after insert") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after insert") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are inserted") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }

  for (size_t &v : values) {
    v *= 2;
  }

  map.batch_insert((size_t) 0,
                   NUM_ELEMS,
                   [](size_t) -> size_t { return 0; },
                   [](size_t i, size_t &state) {
                     const size_t old = state;
                     state++;
                     return std::pair(i, 2 * old);
                   });

  SECTION("size equal to the number of values after updates") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("queues are empty after updates") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }
  SECTION("all values are updated") {
    std::vector<size_t> extracted_values(NUM_ELEMS);
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      if (value != map.end()) {
        extracted_values[i] = value->second;
      } else {
        FAIL("entry for " << i << " does not exist");
      }
    }
    REQUIRE_THAT(extracted_values, Catch::Matchers::UnorderedRangeEquals(values));
  }
}

TEST_CASE("range batch insertions", "[batch] [range]") {

  using Map = ShardedMap<char, CharMetric, std::unordered_map, FindChar>;
  Map map(NUM_THREADS, 128);

  static_assert(std::ranges::random_access_range<std::string>);

  // Seed random number generator
  std::srand(std::time(nullptr));

  // Fill a string with 100000 random chars from 'a' to 'z'
  std::string s;
  s.resize(NUM_ELEMS);
  std::generate(s.begin(), s.end(), []() { return std::rand() % 26 + 'a'; });

  map.batch_insert(s, [](auto c) { return std::pair{c, (size_t) 0}; });

  SECTION("queues are empty after insert") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }

  SECTION("all values are inserted") {
    size_t num_chars = 0;
    map.for_each([&num_chars](const char &, const CharMetric &cm) { num_chars += cm.count; });
    REQUIRE(num_chars == s.size());
  }

  map.batch_insert(s, [](auto &c) { return std::pair{c, (size_t) 0}; });

  SECTION("queues are empty after updates") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());
    REQUIRE(max_load == 0);
  }

  SECTION("all values are updated") {
    size_t num_chars = 0;
    map.for_each([&num_chars](const char &, const CharMetric &cm) { num_chars += cm.count; });
    REQUIRE(num_chars == 2 * s.size());
  }
}

TEST_CASE("range batch state insertions", "[batch] [range] [state]") {

  using Map = ShardedMap<char, CharMetric, std::unordered_map, FindChar>;
  Map map(NUM_THREADS, 128);

  static_assert(std::ranges::random_access_range<std::string>);

  // Seed random number generator
  std::srand(std::time(nullptr));

  std::string s;
  s.resize(NUM_ELEMS);
  std::generate(s.begin(), s.end(), []() { return std::rand() % 26 + 'a'; });
  std::vector<size_t> char_counters = map.batch_insert(
      s,
      [](size_t) { return (size_t) 0; },
      [](char c, size_t &counter) {
        counter++;
        return std::pair(c, (size_t) 0);
      });

  SECTION("queues are empty after insert") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());

    REQUIRE(max_load == 0);
  }

  SECTION("all values are inserted") {
    size_t num_chars = 0;
    map.for_each([&num_chars](const char &, const CharMetric &cm) { num_chars += cm.count; });
    REQUIRE(num_chars == s.size());
  }

  SECTION("insertions tracked by all threads") {
    const size_t num_chars = std::accumulate(char_counters.begin(), char_counters.end(), 0);
    REQUIRE(num_chars == s.size());
  }

  char_counters = map.batch_insert(
      s,
      [](size_t) { return (size_t) 0; },
      [](char c, size_t &counter) {
        counter++;
        return std::pair(c, (size_t) 0);
      });

  SECTION("queues are empty after updates") {
    const auto   loads    = map.queue_loads();
    const size_t max_load = *std::max_element(loads.begin(), loads.end());
    REQUIRE(max_load == 0);
  }

  SECTION("all values are updated") {
    size_t num_chars = 0;
    map.for_each([&num_chars](const char &, const CharMetric &cm) { num_chars += cm.count; });
    REQUIRE(num_chars == 2 * s.size());
  }

  SECTION("updates tracked by all threads") {
    const size_t num_chars = std::accumulate(char_counters.begin(), char_counters.end(), 0);
    REQUIRE(num_chars == s.size());
  }
}
