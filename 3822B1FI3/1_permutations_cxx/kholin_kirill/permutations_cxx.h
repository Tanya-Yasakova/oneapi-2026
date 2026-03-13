#ifndef PERMUTATIONS_CXX_H_
#define PERMUTATIONS_CXX_H_

#include <map>
#include <string>
#include <vector>

using dictionary_t = std::map<std::string, std::vector<std::string>>;

void Permutations(dictionary_t &dictionary);

#endif // PERMUTATIONS_CXX_H_