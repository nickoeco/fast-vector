#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "fast_vector/flat_index.h"
#include "fast_vector/hnsw_index.h"

namespace {
using Clock = fast_vector::SearchContext::Clock;
double us(Clock::duration duration) {
  return std::chrono::duration<double, std::micro>(duration).count();
}
}  // namespace

int main() {
  constexpr std::size_t count = 3000, flat_count = 30000, dimension = 64, repetitions = 20;
  std::mt19937 random(42);
  std::uniform_real_distribution<float> uniform(-1, 1);
  fast_vector::FlatIndex flat(dimension);
  // Deliberately broad traversal makes cooperative cancellation observable on a small fixture.
  fast_vector::HnswIndex hnsw(
      {.dimension = dimension, .max_connections = 8, .ef_construction = 32, .ef_search = count});
  for (std::size_t id = 0; id < flat_count; ++id) {
    std::vector<float> vector(dimension);
    for (auto& value : vector) value = uniform(random);
    flat.add(id, vector);
    if (id < count) hnsw.add(id, vector);
  }
  std::vector<float> query(dimension);
  for (auto& value : query) value = uniform(random);
  std::cout << "index,seed,vectors,dimension,k,ef_search,interval,average_query_us,"
               "cancelled_trials,completed_without_interruption,cancel_to_completion_average_us\n";
  for (const auto* index : std::vector<const fast_vector::VectorIndex*>{&flat, &hnsw}) {
    for (const std::size_t interval : {0U, 1U, 16U, 64U, 256U}) {
      fast_vector::SearchContext context;
      if (interval != 0) {
        context = {Clock::now() + std::chrono::hours(1),
                   std::make_shared<fast_vector::SearchCancellation>(), interval};
      }
      for (int warmup = 0; warmup < 3; ++warmup) (void)index->search(query, 10, context);
      const auto started = Clock::now();
      for (std::size_t trial = 0; trial < repetitions; ++trial)
        (void)index->search(query, 10, context);
      const double average = us(Clock::now() - started) / repetitions;
      unsigned interrupted = 0, completed = 0;
      double cancellation_total = 0;
      if (interval != 0) {
        for (int trial = 0; trial < 10; ++trial) {
          auto cancellation = std::make_shared<fast_vector::SearchCancellation>();
          std::promise<void> entered;
          auto running = entered.get_future();
          auto task = std::async(std::launch::async, [&] {
            entered.set_value();
            try {
              (void)index->search(query, 10,
                                  fast_vector::SearchContext{Clock::now() + std::chrono::hours(1),
                                                             cancellation, interval});
              return false;
            } catch (const fast_vector::SearchInterrupted&) {
              return true;
            }
          });
          running.wait();
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          const auto requested = Clock::now();
          cancellation->request(fast_vector::SearchStopReason::Cancelled);
          if (task.get()) {
            ++interrupted;
            cancellation_total += us(Clock::now() - requested);
          } else
            ++completed;
        }
      }
      std::cout << (index == &flat ? "flat" : "hnsw") << ",42,"
                << (index == &flat ? flat_count : count) << ',' << dimension << ",10,"
                << (index == &flat ? 0 : count) << ',' << interval << ',' << average << ','
                << interrupted << ',' << completed << ',';
      if (interrupted)
        std::cout << cancellation_total / interrupted;
      else
        std::cout << "NA";
      std::cout << '\n';
    }
  }
}
