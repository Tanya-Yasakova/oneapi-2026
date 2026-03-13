#include "permutations_cxx.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> anagram_groups;
  anagram_groups.reserve(dictionary.size());

  for (const auto &entry : dictionary) {
    const std::string &original = entry.first;

    std::string sorted = original;
    std::sort(sorted.begin(), sorted.end());

    anagram_groups[std::move(sorted)].push_back(original);
  }

  for (auto &entry : dictionary) {
    const std::string &original = entry.first;
    std::vector<std::string> &permutations = entry.second;

    std::string sorted = original;
    std::sort(sorted.begin(), sorted.end());

    const auto &group = anagram_groups[sorted];

    permutations.clear();
    permutations.reserve(group.size() - 1);

    for (const auto &candidate : group) {
      if (candidate != original) {
        permutations.push_back(candidate);
      }
    }

    std::sort(permutations.rbegin(), permutations.rend());
  }
}