#ifndef ML_HPP
#define ML_HPP

#include "private/details.hpp"

namespace ml {
/**
 * Sets the number of threads to be used.
 *
 * @param num_threads The number of threads to be set. If `num_threads` is 0,
 * it will be set to 1.
 */
static void setNumThreads(const unsigned int num_threads) {
  _details::num_threads = (num_threads == 0) ? 1 : num_threads;
}

/**
 * Gets the number of threads currently set.
 *
 * @return The number of threads.
 */
static unsigned int getNumThreads() { return _details::num_threads; }

} // namespace ml

#endif //ML_HPP
