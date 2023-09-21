# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import filecmp
import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from tools import mlir_pytaco_api as pt
from tools import testing_utils as utils


import mlir_graphblas as mlgb
from mlir_graphblas import types, tensor, operators, operations
from mlir_graphblas import descriptor as desc


# indices_a = [(0, 1), (1, 0), (1, 2)]
# indices_b = [(0, 2), (1, 0), (1, 2)]

# a = tensor.Matrix.new(types.FP32, 5, 5)
# a.build(*list(zip(*indices_a)), [3, 2, 5], colwise=False, sparsity=["compressed", "compressed"])

# b = tensor.Matrix.new(types.FP32, 5, 5)
# b.build(*list(zip(*indices_b)), [3, 4, 5], colwise=False, sparsity=["compressed", "compressed"])

# out = tensor.Matrix.new(types.FP32, 5, 5)
# operations.ewise_add(out, operators.BinaryOp.plus, a, b)
# print(out.extract_tuples())

def main():
    indices_a = [(0, 1), (1, 0), (1, 2)]
    indices_b = [(0, 2), (1, 0), (1, 2)]

    a = tensor.Matrix.new(types.FP32, 5, 5)
    a.build(*list(zip(*indices_a)), [3, 2, 5], colwise=False, sparsity=["compressed", "compressed"])

    b = tensor.Matrix.new(types.FP32, 5, 5)
    b.build(*list(zip(*indices_b)), [3, 4, 5], colwise=False, sparsity=["compressed", "compressed"])

    out = tensor.Matrix.new(types.FP32, 5, 5)
    operations.ewise_add(out, operators.BinaryOp.plus, a, b)
    print(out.extract_tuples())
    # a = pt.tensor([5, 5], [pt.compressed, pt.compressed])
    # b = pt.tensor([5, 5], [pt.compressed, pt.compressed])
    # c = pt.tensor([a.shape[0], b.shape[1]], [pt.compressed, pt.compressed])
    # c_kokkos = pt.tensor([a.shape[0], b.shape[1]], [pt.compressed, pt.compressed])

    # a.insert([0,1], 3.0)
    # a.insert([1,0], 2.0)
    # a.insert([1,2], 5.0)

    # b.insert([0,2], 3.0)
    # b.insert([1,0], 4.0)
    # b.insert([1,2], 5.0)

    # i, j = pt.get_index_vars(2) 
    # c[i,j] = a[i,j] + b[i,j]
    # c_kokkos[i,j] = a[i,j] + b[i,j]

    
    # golden_file = "gold_c.tns"
    # out_file = "c.tns"
    # pt.write(golden_file, c)
    # pt.write_kokkos(out_file, c_kokkos)


    # operations.ewise_add(out, operators.BinaryOp.plus, a, b)

    #
    # CHECK: Compare result True
    #
    # print(f"Compare result {utils.compare_sparse_tns(golden_file, out_file)}")


if __name__ == "__main__":
    main()
