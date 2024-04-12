#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <ranges>
#include <sharded_map/sharded_map.hpp>
#include <unordered_map>

constexpr size_t NUM_THREADS = 4;

unsigned int Factorial(unsigned int number) { return number <= 1 ? number : Factorial(number - 1) * number; }

TEST_CASE("Factorials are computed", "[factorial]") {
  REQUIRE(Factorial(1) == 1);
  REQUIRE(Factorial(2) == 2);
  REQUIRE(Factorial(3) == 6);
  REQUIRE(Factorial(10) == 3628800);
}

using namespace sharded_map;

TEST_CASE("sharded map basic insertions", "[sharded] [insertions]") {
  using Map = ShardedMap<size_t, size_t, std::unordered_map, update_functions::Overwrite<size_t, size_t>>;
  Map map(NUM_THREADS, 128);

  constexpr size_t NUM_ELEMS = 100000;

  std::vector<int> values(NUM_ELEMS);
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
      shard.insert(i, i);
    }

    // Mark that another thread is done
    threads_done++;

    // We wait for the other threads to be done.
    // As queues might still fill up, each thread handles its queue
    // However, we won't force the other threads to handle their queues
    while (threads_done.load() < NUM_THREADS) {
      shard.handle_queue_sync(false);
    }
    // This thread is completely done and we won't make the other threads wait for this one,
    // if they are still stuck in the loop above
    barrier.arrive_and_drop();
  }

  SECTION("size equal to the number of values after insert") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("all values are inserted") {
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      REQUIRE(value != map.end());
      REQUIRE(value->second == i);
    }
  }

  // calling arrive_and_drop modifies the barrier permanently. see the docs for std::barrier on cppreference
  map.reset_barrier();
  threads_done = 0;

  // We insert the tuples (i, 2*i). Since we used the Overwrite Update function, this should overwrite the values
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
    barrier.arrive_and_drop();
  }

  SECTION("size equal to the number of values after updates") { REQUIRE(map.size() == NUM_ELEMS); }
  SECTION("all values are inserted") {
    for (size_t i = 0; i < NUM_ELEMS; i++) {
      auto value = map.find(i);
      REQUIRE(value != map.end());
      REQUIRE(value->second == 2 * i);
    }
  }
}
