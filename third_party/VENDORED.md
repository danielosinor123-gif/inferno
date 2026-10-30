# Vendored ggml

- Upstream: https://github.com/ggml-org/llama.cpp
- Tag: `v0.8.0`
- Pinned commit: `8642541b255bfb5dc39cd351a178be09013a55d6`
- License: MIT (see `ggml/LICENSE`)
- Vendored: `ggml/src` + `ggml/include` (cross-vendor backend trees —
  cuda/metal/hip/vulkan/etc. — are NOT copied; they remain referenced by the
  pin). A CUDA backend tree is added when G1's GPU slice is enabled.
- Consumed by: `crates/inferno-gpu/build.rs` (static build, CPU backend first).
- Why vendored rather than fetched: the build must work offline on a flaky
  network (the repo's standing lesson), and the pin must be reviewable.
