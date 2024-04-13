
#include <algorithm>
#include <cstdlib>
#include <ctime>

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

// Check that FindChar actually fulfills the "UpdateFunction" concept
// for a sharded map with keys of type char and values of type CharMetric
static_assert(sharded_map::UpdateFunction<FindChar, char, CharMetric>,
              "FindChar does not fulfill UpdateFunction concept");
