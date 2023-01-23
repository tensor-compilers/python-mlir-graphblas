# python-mlir-graphblas
Implementation of the GraphBLAS spec using MLIR python bindings

A full GraphBLAS implementation is possible in relatively few lines
of Python code because we rely heavily on the `linalg.generic` operation
in MLIR, specifically its appropriate handling of different Sparse Tensor
layouts. Each GraphBLAS operation becomes essentially a single
`linalg.generic` operation with some minor pre- and post-handling of
object creation.


# Usage

### Create a Matrix or Vector

```python
>>> from mlir_graphblas import types, operations, operators, Matrix, Vector

>>> m = Matrix.new(types.FP64, 2, 5)
>>> m.build([0, 0, 1, 1, 1], [1, 3, 0, 3, 4], [1., 2., 3., 4., 5.75])
>>> m
Matrix<FP64, shape=(2, 5), format=CSR>

>>> v = Vector.new(types.FP64, 5)
>>> v.build([0, 2, 3, 4], [3., -2., 4., 1.5])
>>> v
Vector<FP64, size=5>
```

### Perform Operations

Each operation requires an output object to be passed in.

```python
>>> from mlir_graphblas.operators import Semiring

>>> z = Vector.new(types.FP64, 2)
>>> operations.mxv(z, Semiring.plus_times, m, v)
>>> z
Vector<FP64, size=2)
```

### View Results

```python
>>> indices, values = z.extract_tuples()
>>> indices
array([0, 1], dtype=uint64)
>>> values
array([ 8.   , 33.625])
```

# Building MLIR

Until LLVM 16.0 is released, the required MLIR operations in the sparse_tensor dialect will only be
available on the main repo branch, meaning there aren't any packages readily available to install
from conda-forge.

To build locally, download the LLVM repo from GitHub.

```
git clone https://github.com/llvm/llvm-project.git
```

Next create a conda environment for building the project.

```
conda create -n mlir_build_env -y
conda activate mlir_build_env
conda install python=3.10 numpy pyyaml cmake ninja pybind11 python-mlir-graphblas
```

Define `PREFIX` as the location of your environment (active env location when running `conda info`).
Then run cmake.

```
export PREFIX=/location/to/your/conda/environment

cd llvm-project
mkdir build
cd build

cmake -G Ninja ../llvm \
   -DCMAKE_INSTALL_PREFIX=$PREFIX \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_BUILD_LLVM_DYLIB=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=`which python`
```

If building on a Mac, perform this additional step.

```
cp $PREFIX/lib/libtinfo* lib/
cp $PREFIX/lib/libz* lib/
```

Then build the project

```
cmake --build .
cmake --install .
```

Finally, set `LLVM_BUILD_DIR` to point to the current build directory.
Now python-mlir-graphblas should be usable. Verify by running tests.

```
LLVM_BUILD_DIR=. pytest --pyargs mlir_graphblas
```
