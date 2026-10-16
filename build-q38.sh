#!/bin/bash
# the qwen38 branch engine, configured like project-maya/build; args: targets (default strata-dense)
cd "$(dirname "$0")" || exit 1
V=/media/per/work/llm/project-maya/.venv/bin
[ -f build/CMakeCache.txt ] || $V/cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$V/ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_COMPILER=/media/per/work/llm/toolchain/ninfer-env/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DGGML_NATIVE=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR=/media/per/work/llm/project-maya/third_party/llama.cpp >/dev/null || exit 1
for t in ${*:-strata-dense}; do
  # (a failed build must not leave the old binary looking fresh: report and exit non-zero)
  if $V/cmake --build build --target $t -j 8 > /tmp/build-q38-$t.log 2>&1; then echo "BUILT $t"
  else grep -E "error|FAILED" /tmp/build-q38-$t.log | head -30; echo "BUILD FAILED $t"; exit 1; fi
done
