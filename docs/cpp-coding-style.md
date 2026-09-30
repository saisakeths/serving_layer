# C++ coding style (serving_layer)

Use this in all Phase 1+ C++ work in this repo.

## Namespace

- After `#include`s in each `.cpp` and `.hpp`, add `using namespace std;` once.
- Do not prefix standard names with `std::` after that (use `vector`, `string`, not `std::vector`).
- Project API lives in `namespace serving { ... }`.

## Function documentation

Immediately before **every** function (free, member, static), add one comment line:

```cpp
// functionName: ReturnType — brief description of what it does.
```

Example:

```cpp
// matmul: void — multiplies A[M,K] by B[K,N] and writes result into out[M,N].
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
```

Keep the comment on a single line.

## Cursor rule

When Agent mode is available, mirror this file into `.cursor/rules/cpp-serving-layer-style.mdc` with `globs: "**/*.{cpp,hpp,h}"`.
