# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

from mlir_graphblas import types, tensor, operators, operations


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


if __name__ == "__main__":
    main()
