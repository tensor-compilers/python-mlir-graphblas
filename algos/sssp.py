# SSSP using MLIR-Python-GraphBLAS-Kokkos
# Adapted from the Python-GraphBLAS implementation at https://python-graphblas.readthedocs.io/en/latest/getting_started/primer.html
from mlir_graphblas import types, tensor, operators, operations
import numpy as np


def main():
    options = "parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical"
    # options = ""

    # Create the graph and starting vector
    start_node = 0

    row_indices = [0, 0, 1, 1, 2]
    col_indices = [1, 2, 2, 3, 3]
    values      = [2.0, 5.0, 1.5, 4.25, 0.5]

    G = tensor.Matrix.new(types.FP64, 4, 4)
    G.build(row_indices, col_indices, values, colwise=False, sparsity=["compressed", "compressed"])
    # G.build(row_indices, col_indices, values, colwise=False, sparsity=["dense", "dense"])

    v = tensor.Vector.new(types.FP64, 4)
    v.build([start_node],[0.0], sparsity=["compressed"])

    # Compute SSSP
    while True:
        w = v.dup(index_instance=0, num_instances=3,options=options)

        # Perform a BFS step using min_plus semiring
        # Accumulate into v using the `min` function
        # v(op.min) << semiring.min_plus(v @ G)
        operations.vxm(v,operators.Semiring.min_plus, v , G, accum=operators.BinaryOp.min, index_instance=1, num_instances=3,options=options)
        
        # The algorithm is converged once v stops changing
        # if v.isequal(w):
        # TODO: checking via numpy my be slower than doing it in mlir
        v_idx,v_data = v.extract_tuples()
        w_idx,w_data = w.extract_tuples()
        if np.array_equal(v_idx, w_idx) \
            and np.array_equal(v_data,w_data):
            break
    print(v.extract_tuples())
    # Expected: v=[0.0, 2.0, 3.5, 4.0]


if __name__ == "__main__":
    main()
