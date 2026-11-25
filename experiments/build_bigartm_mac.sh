#!/bin/sh
# Build the official CLI with project-local dependencies on Apple Silicon.
set -eu
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "This helper targets macOS arm64."; exit 1; }
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
buildroot="$project/.cache/native"
src="$buildroot/src"
prefix="$buildroot/deps"
build="$buildroot/build"
mkdir -p "$buildroot" "$project/.cache/bin"
if [ ! -x "$buildroot/bin/micromamba" ]; then
    curl -fLsS https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o "$buildroot/micromamba.tar.bz2"
    tar -xjf "$buildroot/micromamba.tar.bz2" -C "$buildroot" bin/micromamba
fi
if [ ! -x "$prefix/bin/cmake" ]; then
    "$buildroot/bin/micromamba" create -y --no-rc --root-prefix "$buildroot/mamba" -p "$prefix" -c conda-forge boost-cpp=1.78 cmake=3.31.8 ninja
fi
if [ ! -d "$src/.git" ]; then
    git clone -q https://github.com/bigartm/bigartm.git "$src"
    git -C "$src" checkout --detach 47e37f982de87aa67bfd475ff1f39da696b181b3
fi
# Modern AppleClang requires a const comparator and fixed enum underlying types.
sed -i '' 's/FieldDescriptor\* f2) {/FieldDescriptor* f2) const {/' "$src/3rdparty/protobuf-3.0.0/src/google/protobuf/compiler/java/java_file.cc"
for name in int_float_mixture_enum udt_builtin_mixture_enum; do
    sed -i '' -E "s/enum ${name}([[:space:]]*)$/enum ${name} : int\1/" "$prefix/include/boost/numeric/conversion/$name.hpp"
done
"$prefix/bin/cmake" -S "$src" -B "$build" -G Ninja -DCMAKE_MAKE_PROGRAM="$prefix/bin/ninja" -DCMAKE_PREFIX_PATH="$prefix" "-DCMAKE_CXX_FLAGS=-I$build/src/artm -Wno-enum-constexpr-conversion" -DBUILD_TESTS=OFF -DBUILD_INTERNAL_PYTHON_API=OFF
"$prefix/bin/cmake" --build "$build" --target bigartm -j8
cp "$build/bin/bigartm" "$project/.cache/bin/bigartm"
