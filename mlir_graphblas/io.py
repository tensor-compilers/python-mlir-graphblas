import numpy as np
from scipy.sparse import coo_matrix, coo_array
from scipy.io import mmread, mmwrite
from mlir_graphblas import types, tensor
from typing import Union, Optional


def from_mtx(mtx_fpath: str,sparsity: Optional[list[str]] = ["compressed", "compressed"]) -> tuple[Union[tensor.Matrix,tensor.Vector],tuple[int],int,]:
    # read and build tensor from .mtx matrix market file format
    # tensor_type is either "matrix" or "vector"
    # sparsity is a ndim-sized list of "dense" or "compressed"
    # returns the shape, number of entries, and the matrix or vector
    mtx = mmread(mtx_fpath)
    if not (isinstance(mtx, coo_matrix) or isinstance(mtx, coo_array)):
        raise ValueError
    nrows,ncols = mtx.shape
    dtype = types.DType.from_np(mtx.dtype)
    row_indices = mtx.row.astype(np.uint64)
    col_indices = mtx.col.astype(np.uint64)
    if len(sparsity) == 2:
        # matrix
        mtx_tensor = tensor.Matrix.new(dtype,nrows,ncols)
        mtx_tensor.build(row_indices,col_indices,mtx.data,colwise=False,sparsity=sparsity)
    elif len(sparsity) == 1:
        # vector
        mtx_tensor = tensor.Vector.new(dtype,nrows)
        mtx_tensor.build(row_indices,mtx.data,sparsity=sparsity)
    else:
        raise ValueError
    return mtx_tensor


def to_mtx(fout: str, mtx_tensor: Union[tensor.Matrix,tensor.Vector],shape=[]) -> None:
    # write mlir matrix, vector, or scalar to matrix market file format .mtx
    if mtx_tensor.ndims == 0:  # scalar
        mtx = coo_array((mtx_tensor.extract_element(),(0, 0)), shape=(1,1))
        mmwrite(fout,mtx,field="real", symmetry="general")
    if mtx_tensor.ndims == 2:  # matrix
        row_indices,col_indices,data = mtx_tensor.extract_tuples()
        if not shape:
            num_rows = mtx_tensor.nrows()
            num_cols = mtx_tensor.ncols()
        else:
            num_rows = shape[0]
            num_cols = shape[1]
        mtx = coo_array((data,(row_indices,col_indices)), shape=(num_rows, num_cols))
        mmwrite(fout, mtx, field="real", symmetry="general")
    elif mtx_tensor.ndims == 1:  # vector
        row_indices,data = mtx_tensor.extract_tuples()
        if not shape:
            num_elems = mtx_tensor.size()
        else:
            num_elems = shape[0]
        mtx = coo_array((data,(row_indices, np.zeros(num_elems))), shape=(num_elems,1))
        mmwrite(fout, mtx, field="real", symmetry="general")


def to_dense_np(mlir_tensor):
    if mlir_tensor.ndims == 2:
        row_indices,col_indices,data = mlir_tensor.extract_tuples()
        np_mtx = coo_array((data,(row_indices,col_indices)), shape=(7,7)).toarray()
        return np_mtx
    elif mlir_tensor.ndims == 1:
        row_indices,data = mlir_tensor.extract_tuples()
        np_vec = coo_array((data,(row_indices, np.zeros(len(row_indices)))), shape=(7,1)).toarray()
        return np_vec
    else:
        raise ValueError
