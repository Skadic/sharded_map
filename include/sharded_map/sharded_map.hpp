#pragma once

#include <algorithm>
#include <atomic>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <omp.h>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace sharded_map {

namespace util {

/// @brief A mixing function to provide better avalanching to intermediate hash
/// values.
///
/// https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
constexpr uint64_t mix_select(uint64_t key) {
  key ^= (key >> 31);
  key *= 0x7fb5d329728ea185;
  key ^= (key >> 27);
  key *= 0x81dadef4bc2dd44d;
  key ^= (key >> 33);
  return key;
}

/// @brief Returns the ceiling of x / y for x > 0;
///
/// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
constexpr size_t ceil_div(std::integral auto x, std::integral auto y) {
  return 1 + (static_cast<size_t>(x) - 1) / static_cast<size_t>(y);
}

} // namespace util

/// @brief Represents an update function which given a value,
/// updates a value in the map.
///
/// @tparam Fn The type of the update function.
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
///
template<typename Fn, typename K, typename V>
concept UpdateFunction = requires(const K &k, V &v_lv, typename Fn::InputValue in_v_rv) {
  typename Fn::InputValue;
  // Updates a pre-existing value in the map.
  // Arguments are the key, the value in the map,
  // and the input value used to update the value in
  // the map
  { Fn::update(k, v_lv, std::move(in_v_rv)) } -> std::same_as<void>;
  // Initialize a value from an input value
  // Arguments are the key, and the value used to
  // initialize the value in the map. This returns the
  // value to be inserted into the map
  { Fn::init(k, std::move(in_v_rv)) } -> std::convertible_to<V>;
};

template<typename Fn>
concept StateCreatorFunction =
    std::invocable<size_t> && !std::is_void_v<std::result_of_t<Fn(size_t)>>;

template<typename Fn, typename K, typename V, typename State>
concept StatefulGeneratorFunction =
    !std::is_void_v<State> && std::invocable<Fn, size_t, std::add_lvalue_reference_t<State>> &&
    std::same_as<std::result_of_t<Fn(size_t, std::add_lvalue_reference_t<State>)>, std::pair<K, V>>;

template<typename Fn, typename K, typename V>
concept StatelessGeneratorFunction =
    std::invocable<Fn, size_t> && std::same_as<std::result_of_t<Fn(size_t)>, std::pair<K, V>>;

template<typename Fn, typename K, typename V, typename State>
concept PairGeneratorFunction =
    StatefulGeneratorFunction<Fn, K, V, State> || StatelessGeneratorFunction<Fn, K, V>;

namespace update_functions {
///
/// @brief An update function for sharded maps which on update just overwrites
/// the value.
///
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
///
template<std::copy_constructible K, std::move_constructible V>
struct Overwrite {
  using InputValue = V;

  inline static void update(const K &, V &value, V &&input_value) { value = input_value; }

  inline static V init(const K &, V &&input_value) { return input_value; }
};

///
/// @brief An update function for sharded maps which upon update does nothing.
///   If the value does not exist, the value is inserted.
///
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
///
template<std::copy_constructible K, std::move_constructible V>
struct Keep {
  using InputValue = V;
  inline static void update(const K &, V &, V &&) {}

  inline static V init(const K &, V &&input_value) { return input_value; }
};

} // namespace update_functions

/// @brief A hash map that must be used by multiple threads, each thread having only having write
/// access to a certain segment of the hash input space.
///
/// Each processor has one queue and one local hash map.
///
/// To use this map with `p` threads, create a shard for each thread using `get_shard` with thread
/// id's ranging from 0 to (inclusively) p-1. Each possible hash has exactly one processor that is
/// responsible for storing that hash. Inserting an element inserts it into its responsible
/// processor's queue. `handle_queue_sync` and `handle_queue_async` empty the processor's queue and
/// insert/update all values in the local hash map. `handle_queue_sync` is automatically called by a
/// processor trying to insert a value, if the responsible thread's queue is full. So, if all
/// threads are continuously inserting elements, it should suffice for all processors to keep
/// calling `insert` without manually handling the queues.
///
/// If not all processors are inserting elements at some point, then it becomes necessary for
/// processors to handle the queues manually using any of the two `handle_queue` methods.
///
/// @tparam K The type of the keys in the hash map.
/// @tparam V The type of the values in the hash map.
/// @tparam SeqHashMapType The type of the hash map used internally. This should be compatible with
/// std::unordered_map.
/// @tparam UpdateFn The update function deciding how to insert or update values in the map.
template<std::copy_constructible K,
         std::move_constructible V,
         template<typename, typename, typename...> typename SeqHashMapType = std::unordered_map,
         UpdateFunction<K, V> UpdateFn = update_functions::Overwrite<K, V>>
  requires std::movable<typename UpdateFn::InputValue>
class ShardedMap {
  /// The sequential backing hash map type
  using SeqHashMap = SeqHashMapType<K, V>;

  /// The sequential hash map's hasher
  using Hasher = typename SeqHashMap::hasher;

  /// The type used for updates
  using InputValue = typename UpdateFn::InputValue;

  /// The actual pair of key and value stored in the queues
  using QueueStoredValue = std::pair<K, InputValue>;

  /// The MPSC queue type
  using Queue = std::span<QueueStoredValue>;

  /// The memory order namespace from the standard library
  using mem = std::memory_order;

  /// @brief The number of threads operating on this map.
  const size_t thread_count_;

  /// @brief Contains a hash map for each thread
  std::vector<SeqHashMap> map_;

  /// @brief Contains a task queue for each thread, holding insert
  ///   operations for each thread.
  std::vector<Queue> task_queue_;

  /// @brief Contains the number of tasks in each thread's queue.
  std::span<std::atomic_size_t> task_count_;

  /// @brief Contains the number of threads currently handling their queues.
  ///   This is used
  ///   1. as a signal to other threads that they should handle their queue if >0, and
  ///   2. to keep track of whether all threads have handled their queues.
  std::atomic_size_t threads_handling_queue_;

  /// @brief The maximum number of elements that fit in a queue
  const size_t queue_capacity_;

  /// @brief The function to run when a barrier is unblocked.
  constexpr static std::invocable auto FN = []() noexcept {};

  /// @brief Thread barrier to synchronize threads when handling queues.
  std::unique_ptr<std::barrier<decltype(FN)>> barrier_;

public:
  /// @brief Creates a new sharded map.
  ///
  /// @param thread_count The exact number of threads working on this map. Thread ids must be 0, 1,
  /// ..., thread_count-1.
  /// @param queue_capacity The maximum amount of elements allowed in each queue.
  ///
  ShardedMap(size_t thread_count, size_t queue_capacity) :
      thread_count_(thread_count),
      map_(),
      task_queue_(),
      task_count_(),
      threads_handling_queue_(0),
      queue_capacity_(queue_capacity),
      barrier_(
          std::make_unique<std::barrier<decltype(FN)>>(static_cast<ptrdiff_t>(thread_count), FN)) {
    map_.reserve(thread_count);
    task_queue_.reserve(thread_count);
    task_count_ = std::span<std::atomic_size_t>(new std::atomic_size_t[thread_count], thread_count);
    for (size_t i = 0; i < thread_count; i++) {
      map_.emplace_back();
      task_queue_.emplace_back(new QueueStoredValue[queue_capacity], queue_capacity);
      task_count_[i] = 0;
    }
  }

  ~ShardedMap() {
    delete[] task_count_.data();
    for (auto &queue : task_queue_) {
      delete[] queue.data();
    }
  }

  void reset_barrier() {
    barrier_.reset(new std::barrier<decltype(FN)>{static_cast<ptrdiff_t>(thread_count_), FN});
  }

  class Shard {
    // @brief The sharded map this shard belongs to.
    ShardedMap &sharded_map_;
    // @brief This thread's id.
    const size_t thread_id_;
    /// @brief This shard's local hash map.
    SeqHashMap &map_;
    /// @brief This shard's queue.
    Queue &task_queue_;
    /// @brief The number of elements in this shard's queue.
    std::atomic_size_t &task_count_;

  public:
    ///
    /// @brief Create a new shard for the given thread.
    ///
    /// @param sharded_map The sharded map to which the shard belongs.
    /// @param thread_id The thread's id. Must be less than `thread_count` in the sharded map.
    ///
    Shard(ShardedMap &sharded_map, size_t thread_id) :
        sharded_map_(sharded_map),
        thread_id_(thread_id),
        map_(sharded_map_.map_[thread_id]),
        task_queue_(sharded_map_.task_queue_[thread_id]),
        task_count_(sharded_map.task_count_[thread_id]) {}

    /// @brief Inserts or updates a new value in the map, depending on whether
    /// @param k The key to insert or update a value for.
    /// @param in_value The value with which to insert or update.
    inline void insert_or_update_direct(const K &k, InputValue &&in_value) {
      auto res = map_.find(k);
      if (res == map_.end()) {
        // If the value does not exist, insert it
        K key     = k;
        V initial = UpdateFn::init(key, std::move(in_value));
        map_.emplace(key, std::move(initial));
      } else {
        // Otherwise, update it.
        V &val = res->second;
        UpdateFn::update(k, val, std::move(in_value));
      }
    }

    ///
    /// @brief Handles this thread's queue synchronously with other threads, inserting or updating
    /// all values in its queue.
    ///
    /// This thread will wait until all other threads also call this method before starting to
    /// handle the queues.
    ///
    /// @param cause_wait If true, this will force other threads to handle their queues when they
    /// call insert.
    ///
    void handle_queue_sync(bool cause_wait = true) {
      // If we want to cause other threads to wait, we increment the number of threads handling the
      // queue This will cause other threads to wait when they call insert if >0
      if (cause_wait) {
        sharded_map_.threads_handling_queue_.fetch_add(1, mem::acq_rel);
      }
      sharded_map_.barrier_->arrive_and_wait();

      handle_queue_async();

      if (cause_wait) {
        sharded_map_.threads_handling_queue_.fetch_sub(1, mem::acq_rel);
      }
      sharded_map_.barrier_->arrive_and_wait();
    }

    ///
    /// @brief Handles this thread's queue, inserting or updating all values in its queue.
    ///
    /// Warning: This should not be called while other threads are inserting into this thread's
    /// queue!
    ///
    void handle_queue_async() {
      const size_t num_tasks_uncapped = task_count_.exchange(0, mem::acq_rel);
      const size_t num_tasks          = std::min(num_tasks_uncapped, sharded_map_.queue_capacity_);
      if (num_tasks == 0) {
        return;
      }

      //  Handle all tasks in the queue
      for (size_t i = 0; i < num_tasks; ++i) {
        auto &entry = task_queue_[i];
        insert_or_update_direct(entry.first, std::move(entry.second));
      }
    }

    /// @brief Inserts or updates a new value in the map.
    ///
    /// If the value is inserted into the current thread's map, it is inserted immediately. If not,
    /// then it is added to that thread's queue. It will only be inserted into the map, once the
    /// thread comes around to handle its queue using the handle_queue method.
    ///
    /// @param pair The key-value pair to insert or update.
    void insert(QueueStoredValue &&pair) {
      if (sharded_map_.threads_handling_queue_.load(mem::acquire) > 0) {
        handle_queue_sync();
      }
      const size_t hash             = Hasher{}(pair.first);
      const size_t target_thread_id = util::mix_select(hash) % sharded_map_.thread_count_;

      // Otherwise enqueue the new value in the target thread
      std::atomic_size_t &target_task_count = sharded_map_.task_count_[target_thread_id];

      size_t task_idx = target_task_count.fetch_add(1, mem::acq_rel);
      // If the target queue is full, signal to the other threads, that they
      // need to handle their queue and handle this thread's queue
      if (task_idx >= sharded_map_.queue_capacity_) {
        //  Since we incremented that thread's task count, but didn't insert
        //  anything, we need to decrement it again so that it has the correct
        //  value
        target_task_count.fetch_sub(1, mem::acq_rel);
        handle_queue_sync();
        // Since the queue was handled, the task count is now 0
        insert(std::move(pair));
        return;
      }
      // Insert the value into the queue
      sharded_map_.task_queue_[target_thread_id][task_idx] = std::move(pair);
    }

    /// @brief Inserts or updates a new value in the map.
    ///
    /// If the value is inserted into the current thread's map, it is inserted immediately. If not,
    /// then it is added to that thread's queue. It will only be inserted into the map, once the
    /// thread comes around to handle its queue using the handle_queue method.
    ///
    /// @param key The key of the value to insert.
    /// @param value The value to associate with the key.
    inline void insert(const K &key, InputValue value) { insert(QueueStoredValue(key, value)); }
  };

  ///
  /// @brief Create a shard for the given thread.
  ///
  /// @param thread_id The id of the thread for which to create a shard.
  /// @return A shard for the given thread id.
  ///
  Shard get_shard(const size_t thread_id) { return Shard(*this, thread_id); }

  /// @brief Returns the number of key-value pairs in the map.
  ///
  /// Note, that this method calculates the size for each map separately and
  ///     is therefore not O(1).
  /// @return The number of key-value pairs in the map.
  [[nodiscard]] size_t size() const {
    size_t size = 0;
    for (const SeqHashMap &map : map_) {
      size += map.size();
    }
    return size;
  }

  ///
  /// @brief Batch inserts several elements in parallel with a thread-local state.
  ///
  /// The inserted elements are generated by a generator function
  /// which takes as arguments the current iteration index and a reference to the thread-local
  /// state.
  ///
  /// This function allows creating mutable thread-local state instantes of any type
  /// which are passed to each invocation of the generator function.
  ///
  /// @param start The start index if the iteration.
  /// @param end The (exclusive) end index of the iteration
  /// @param gen_state A function that takes a size_t as input which is the thread id, and returns
  ///   the initial thread-local state of this thread's invocations of the generator function.
  ///   This state may be of any desired type.
  /// @param generate_next A function that generates key-value pairs to insert into the map. It
  ///   takes the current index of the iteration and a reference to the current thread-local state
  ///   and returns an std::pair<K,InputValue>, which is inserted into the map.
  ///
  template<std::invocable<size_t> CreateStateFn,
           std::copyable          State = std::result_of_t<CreateStateFn(size_t)>,
           StatefulGeneratorFunction<K, InputValue, State> GeneratorFn>
  std::vector<State>
  batch_insert(size_t start, size_t end, CreateStateFn gen_state, GeneratorFn generate_next) {
    // Create thread-local state
    std::vector<State> state;
    state.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; i++) {
      state.push_back(gen_state(i));
    }
    if (start >= end) {
      return state;
    }

    const size_t range_size   = end - start;
    const size_t segment_size = util::ceil_div(range_size, thread_count_);

    std::atomic_size_t threads_done;

#pragma omp parallel num_threads(thread_count_)
    {
      const size_t thread_id    = omp_get_thread_num();
      const size_t thread_start = start + thread_id * segment_size;
      const size_t thread_end   = std::min(start + (thread_id + 1) * segment_size, end);
      Shard        shard        = get_shard(thread_id);
      State       &local_state  = state[thread_id];

      for (size_t i = thread_start; i < thread_end; i++) {
        std::pair<K, InputValue> res = generate_next(i, local_state);
        shard.insert(res.first, res.second);
      }

      threads_done++;

      while (threads_done.load() < thread_count_) {
        shard.handle_queue_sync(false);
      }
      shard.handle_queue_async();
      barrier_->arrive_and_drop();
    }

    reset_barrier();
    return state;
  }

  ///
  /// @brief Batch inserts several elements in parallel with a thread-local state.
  ///
  /// The inserted elements are generated by a generator function
  /// which takes as arguments the current iteration index and a reference to the thread-local
  /// state.
  ///
  /// This function allows creating mutable thread-local state instantes of any type
  /// which are passed to each invocation of the generator function.
  ///
  /// @param start The start index if the iteration.
  /// @param end The (exclusive) end index of the iteration
  /// @param gen_state A function that takes a size_t as input which is the thread id, and returns
  ///   the initial thread-local state of this thread's invocations of the generator function. This
  ///   state may be of any desired type.
  /// @param generate_next A function that generates key-value pairs to insert into the map. It
  ///   takes the current index of the iteration and a reference to the current thread-local state
  ///   and returns an std::pair<K,InputValue>, which is inserted into the map.
  ///
  template<StatelessGeneratorFunction<K, InputValue> GeneratorFn>
  void batch_insert(size_t start, size_t end, GeneratorFn generate_next) {
    batch_insert(
        start,
        end,
        [](size_t) -> int { return 0; },
        [&](size_t i, int &) { return generate_next(i); });
  }

  /// @brief The whereabouts of a value in the sharded hash map.
  ///
  /// The three variants denote, whether an element does not exist, is contained in a local hash map
  /// or is still inside a queue.
  enum class Whereabouts { NOWHERE, IN_MAP, IN_QUEUE };

  ///
  /// @brief Searches a hash map and queue to determine whether the given key exists in the sharded
  /// hash table.
  ///
  /// Note, that this is not efficient and only for debug purposes. The time complexity is linear in
  /// the queue size.
  ///
  /// @param k The key to search for.
  /// @return NOWHERE, if the element does not exist. IN_MAP, if the element exists in a hash map.
  /// IN_QUEUE, if the element exists in a queue.
  ///
  [[maybe_unused]] Whereabouts where(const K &k) {
    const size_t                  hash             = Hasher{}(k);
    const size_t                  target_thread_id = util::mix_select(hash) % thread_count_;
    SeqHashMap                   &map              = map_[target_thread_id];
    typename SeqHashMap::iterator it               = map.find(k);
    if (it != map.end()) {
      return Whereabouts::IN_MAP;
    }
    Queue &queue = task_queue_[target_thread_id];
    for (size_t i = 0; i < task_count_[target_thread_id]; ++i) {
      if (queue[i].first == k) {
        return Whereabouts::IN_QUEUE;
      }
    }
    return Whereabouts::NOWHERE;
  }

  /// @brief Runs a method for each value in the map.
  ///
  /// The given function must take const references to a key and a value respectively.
  /// @param f The function or lambda to run for each value.
  void for_each(std::invocable<const K &, const V &> auto f) const {
    for (const SeqHashMap &map : map_) {
      for (const auto &[k, v] : map) {
        f(k, v);
      }
    }
  }

  typename SeqHashMap::iterator end() { return map_.back().end(); }

  typename SeqHashMap::iterator find(const K &key) {
    const size_t                  hash             = Hasher{}(key);
    const size_t                  target_thread_id = util::mix_select(hash) % thread_count_;
    SeqHashMap                   &map              = map_[target_thread_id];
    typename SeqHashMap::iterator it               = map.find(key);
    if (it == map.end()) {
      return end();
    }
    return it;
  }

  /// @brief Returns the number of elements in each queue.
  [[nodiscard]] std::vector<size_t> queue_loads() const {
    std::vector<size_t> loads;
    loads.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i) {
      loads.push_back(task_count_[i].load());
    }
    return loads;
  }

  /// @brief Returns the number of elements in each map.
  [[nodiscard]] std::vector<size_t> map_loads() const {
    std::vector<size_t> loads;
    loads.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i) {
      loads.push_back(map_[i].size());
    }
    return loads;
  }

  /// @brief Return a reference to the thread barrier used by this map.
  std::barrier<decltype(FN)> &barrier() { return *barrier_; }

}; // namespace pasta

} // namespace sharded_map
