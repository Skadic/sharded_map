sharded_map
====================

`sharded_map` provides a sharded hash map implementation for parallel processing. This implementation centers around the :cpp:class:`ShardedMap <sharded_map::ShardedMap>` class.


An example might look like this:

.. code-block:: cpp

  #include <sharded_map/sharded_map.hpp>
  #include <iostream>

  using namespace sharded_map;

  // An update function that tracks the occurrences of characters in a text
  struct FindOccs {
    // Input is text positions
    using InputValue = size_t;

    // We initialize a new value with an empty vector only containing the position found
    static std::vector<size_t> init(const char &, size_t &&new_pos) { return {new_pos}; }

    // On update, we add the new position to the vector
    static void update(const char &, std::vector<size_t> &val_in_map, size_t &&new_pos) {
      val_in_map.push_back(new_pos);
    }
  };

  // FindOccs is an update function for a ShardedMap<char, std::vector<size_t>, ...>
  static_assert(UpdateFunction<FindOccs, char, std::vector<size_t>>);

  int main() {
    // Fill a string with 100000 random chars from 'a' to 'z'
    std::string s = "Hello! How are you doing?";

    // Type definition for our map
    using Map = ShardedMap<char, std::vector<size_t>, std::unordered_map, FindOccs>;

    // Create a map for 4 threads with a queue capacity if 128 elements.
    Map map(4, 128);

    // Iterate from 0 (inclusive) to s.size() (exclusive)
    // The i in the lambda is the current index in the iteration.
    // One invocation of the lambda creates a single key-value pair that is inserted into the map.
    map.batch_insert(0, s.size(), [&s](size_t i) { return std::pair(s[i], i); });

    // Iterate over the map. and execute a function
    map.for_each([](const char &k, const std::vector<size_t> &v) {
      std::cout << k << " appears " << v.size() << " times and first at position "
                << *std::ranges::min_element(v) << std::endl;
    });

    return 0;
  }

For more extensive examples, check the `examples` directory in the repo.


.. toctree::
   :maxdepth: 2

   self
   api/library_root
