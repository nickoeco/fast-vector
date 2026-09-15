#include <iostream>
#include <vector>

#include "fast_vector/flat_index.h"

int main() {
    fast_vector::FlatIndex index(3);
    index.add(101, std::vector<float>{1.0F, 0.0F, 0.0F});
    index.add(102, std::vector<float>{0.8F, 0.2F, 0.0F});
    index.add(103, std::vector<float>{0.0F, 1.0F, 0.0F});

    const std::vector<float> query{1.0F, 0.0F, 0.0F};
    const auto results = index.search(query, 2);

    std::cout << "Top results:\n";
    for (const auto& result : results) {
        std::cout << "id=" << result.id << " score=" << result.score << '\n';
    }
    return 0;
}
