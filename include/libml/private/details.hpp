#ifndef DETAILS_H
#define DETAILS_H

#include <thread>

namespace ml::_details {
static unsigned int num_threads{std::thread::hardware_concurrency()};
} // namespace ml::_details

#endif // DETAILS_H
