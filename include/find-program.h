#pragma once

#include <vector>

#include "dsl/ast.h"
#include "attribute.h"
#include "example-generator.h"

std::experimental::optional<dsl::Program> dfs(size_t max_length, const Attribute &attr, const std::vector<Example> &examples);
std::experimental::optional<dsl::Program> sort_and_add(size_t max_length, const Attribute &attr, const std::vector<Example> &examples);