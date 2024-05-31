# Triangle Counting using MLIR-Python-GraphBLAS-Kokkos
# Sandia algo : ntri = sum (sum ((L * L) .* L))
from mlir_graphblas import types, tensor, operators, operations
from mlir_graphblas import descriptor as desc
import numpy as np
from scipy.sparse import coo_array
""" From python graphblas-algorithms
def total_triangles(G):
    # We use SandiaDot method, because it's usually the fastest on large graphs.
    # For smaller graphs, Sandia method is usually faster: plus_pair(L @ L).new(mask=L.S)
    L, U = G.get_properties("L- U-")
    return plus_pair(L @ U.T).new(mask=L.S).reduce_scalar().get(0)
"""


def LUT(A,options=""):
    # C<L.s> = mxm(L,U',semiring=plus_pair)
    num_rows,num_cols = A.shape
    L = tensor.Matrix.new(types.FP64, num_rows, num_cols)
    operations.select(L, operators.SelectOp.tril, A, -1, index_instance=1, num_instances=5,options=options)

    U = tensor.Matrix.new(types.FP64, num_rows, num_cols)
    operations.select(U, operators.SelectOp.triu, A, 1, index_instance=2, num_instances=5,options=options)

    C = tensor.Matrix.new(types.FP64, num_rows, num_cols)
    operations.mxm(C,operators.Semiring.plus_pair, L, U, mask=L, desc=desc.ST1, index_instance=3, num_instances=5,options=options)

    n_tri = tensor.Scalar.new(types.INT32)
    operations.reduce_to_scalar(n_tri, operators.Monoid.plus, C, index_instance=5, num_instances=5,options=options)

    return n_tri.extract_element()


def LL(A, options=""):
    # C<L.s> = mxm(L,L,semiring=plus_pair)
    num_rows,num_cols = A.shape
    L = tensor.Matrix.new(types.FP64, num_rows, num_cols)
    operations.select(L, operators.SelectOp.tril, A, -1, index_instance=1, num_instances=5,options=options)

    C = tensor.Matrix.new(types.FP64, num_rows, num_cols)
    # operations.mxm(C,operators.Semiring.plus_pair, L, L, mask=L,index_instance=2, num_instances=5,options=options)
    operations.mxm(C,operators.Semiring.plus_pair, L, L, mask=L, desc=desc.S, index_instance=2, num_instances=5,options=options)

    n_tri = tensor.Scalar.new(types.INT32)
    operations.reduce_to_scalar(n_tri, operators.Monoid.plus, C, index_instance=5, num_instances=5,options=options)
    print("Number of Triangles =", n_tri.extract_element())

def random_graph(num_nodes, sparsity=0.1):
    import networkx as nx
    G = nx.fast_gnp_random_graph(num_nodes,sparsity)
    triangles = nx.triangles(G)  # {node:tri count}
    A = nx.to_scipy_sparse_array(G, nodelist=None, dtype=np.float64, weight=None, format='coo')
    nrows,ncols = A.shape
    dtype = types.DType.from_np(A.dtype)
    row_indices = A.row.astype(np.uint64)
    col_indices = A.col.astype(np.uint64)
    mtx_tensor = tensor.Matrix.new(dtype,nrows,ncols)
    mtx_tensor.build(row_indices,col_indices,A.data,colwise=False,sparsity=sparsity)
    return triangles, mtx_tensor

def main():
    # options = "parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical"
    options = ""

    # # Matrix from https://zenodo.org/records/4318870
    # row_indices = [1,1,2,2,2,2,3,3,3,4,4,4,4,4,5,5,5,6,6,6,7,7,7,7]
    # col_indices = [2,4,1,4,5,7,4,6,7,1,2,3,6,7,2,6,7,3,4,5,2,3,4,5]
    # values      = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # row_indices = list([x-1 for x in row_indices])  # zero-index
    # col_indices = list([x-1 for x in col_indices])  # zero-index

    # # np_matrix = coo_array((values,(row_indices,col_indices)), shape=(7  , 7)).toarray()

    # A = tensor.Matrix.new(types.FP64, 7, 7)
    # A.build(row_indices, col_indices, values, colwise=False, sparsity=["compressed", "compressed"])

    num_tris = LUT(A,options)
    print("Number of Triangles =", num_tris)


if __name__ == "__main__":
    main()
