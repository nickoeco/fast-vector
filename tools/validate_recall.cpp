#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fast_vector/flat_index.h"

namespace {

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        const std::size_t comma = line.find(',', begin);
        fields.push_back(line.substr(begin, comma - begin));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return fields;
}

fast_vector::VectorId parse_id(const std::string_view text) {
    fast_vector::VectorId value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("invalid integer in CSV: " + std::string(text));
    }
    return value;
}

float parse_float(const std::string& text) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error("invalid float in CSV: " + text);
    }
    return value;
}

std::vector<float> parse_components(
    const std::vector<std::string>& fields, const std::size_t dimension) {
    if (fields.size() != dimension + 1) {
        throw std::runtime_error("CSV row has an unexpected number of columns");
    }
    std::vector<float> values;
    values.reserve(dimension);
    for (std::size_t i = 1; i < fields.size(); ++i) {
        values.push_back(parse_float(fields[i]));
    }
    return values;
}

std::size_t dimension_from_header(const std::string& header) {
    const auto fields = split_csv_row(header);
    if (fields.size() < 2) {
        throw std::runtime_error("CSV must contain an ID column and at least one component");
    }
    return fields.size() - 1;
}

fast_vector::FlatIndex load_vectors(
    const std::string& path, const fast_vector::DotProductKernel kernel) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open vector CSV: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("vector CSV is empty");
    }
    const std::size_t dimension = dimension_from_header(line);
    fast_vector::FlatIndex index(dimension, kernel);

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv_row(line);
        index.add(parse_id(fields.front()), parse_components(fields, dimension));
    }
    return index;
}

std::vector<std::pair<std::uint64_t, std::vector<float>>> load_queries(
    const std::string& path, const std::size_t dimension) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open query CSV: " + path);
    }

    std::string line;
    if (!std::getline(input, line) || dimension_from_header(line) != dimension) {
        throw std::runtime_error("query CSV dimension does not match the index");
    }

    std::vector<std::pair<std::uint64_t, std::vector<float>>> queries;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv_row(line);
        queries.emplace_back(parse_id(fields.front()), parse_components(fields, dimension));
    }
    return queries;
}

std::size_t parse_k(const std::string_view text) {
    std::size_t k = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), k);
    if (error != std::errc{} || end != text.data() + text.size() || k == 0) {
        throw std::invalid_argument("k must be a positive integer");
    }
    return k;
}

fast_vector::DotProductKernel parse_kernel(const std::string_view text) {
    if (text == "scalar") {
        return fast_vector::DotProductKernel::Scalar;
    }
    if (text == "auto") {
        return fast_vector::DotProductKernel::AutoVectorized;
    }
    if (text == "avx2" && fast_vector::avx2_dot_product_available()) {
        return fast_vector::DotProductKernel::Avx2;
    }
    throw std::invalid_argument("kernel must be scalar, auto, or an available avx2");
}

}  // namespace

int main(const int argc, char* argv[]) {
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: fast_vector_validate <vectors.csv> <queries.csv> <k> "
                     "<results.csv> [scalar|auto|avx2]\n";
        return 2;
    }

    try {
        const auto kernel = argc == 6 ? parse_kernel(argv[5])
                                      : fast_vector::DotProductKernel::Scalar;
        const fast_vector::FlatIndex index = load_vectors(argv[1], kernel);
        const auto queries = load_queries(argv[2], index.dimension());
        const std::size_t k = parse_k(argv[3]);

        std::ofstream output(argv[4]);
        if (!output) {
            throw std::runtime_error("cannot open result CSV: " + std::string(argv[4]));
        }
        output << "query_id,rank,vector_id,score\n";
        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (const auto& [query_id, query] : queries) {
            const auto results = index.search(query, k);
            for (std::size_t rank = 0; rank < results.size(); ++rank) {
                output << query_id << ',' << rank << ',' << results[rank].id << ','
                       << results[rank].score << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Validation runner failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
