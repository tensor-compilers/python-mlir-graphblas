# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

from mlir_graphblas import types, tensor, operators, operations


def main():
    indices_a = [(0, 1), (1, 0), (1, 2)]
    indices_b = [0, 1, 2]

    A = tensor.Matrix.new(types.FP32, 5, 5)
    A.build(*list(zip(*indices_a)), [3, 2, 5], colwise=False, sparsity=["compressed", "compressed"])

    b = tensor.Vector.new(types.FP32, 5)
    b.build(indices_b, [3, 4, 5], sparsity=["dense"])

    c = tensor.Vector.new(types.FP32, 5)

    operations.mxv(c, operators.Semiring.plus_times, A, b)
    print(c.extract_tuples())
    

if __name__ == "__main__":
    main()
