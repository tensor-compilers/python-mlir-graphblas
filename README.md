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
