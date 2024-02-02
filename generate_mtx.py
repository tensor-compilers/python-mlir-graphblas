import argparse
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import random


# supported dtypes
dtypes = {"INT8":np.int8,
          "INT16":np.int16,
          "INT32":np.int32,
          "INT64":np.int64,
          "FP32":np.float32,
          "FP64":np.float64}


def generate_random_sparse_mtx(fname, nrows, ncols, density=0.01, dtype=np.float64, seed=42):
    # creates a random sparse matrix
    if np.issubdtype(dtype, np.integer):
        field = "integer"
    else:
        field = "real"
    mtx = random(nrows, ncols, density=density, format='coo', dtype=dtype, random_state=seed, data_rvs=None)
    print(f"Generated matrix shape={mtx.shape} NNZ={mtx.nnz} DType={mtx.dtype} successfully!")
    mmwrite(fname, mtx, field=field, symmetry="general")
    print(f"Matrix saved to {fname}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(prog="Random Matrix Generator", description="Generates a random sparse matrix in Matrix Market file format")
    parser.add_argument("fout", help="Path to save generated matrix in Matrix Market file format")
    parser.add_argument("nrows", type=int, help="Number of rows")
    parser.add_argument("ncols", type=int, help="Number of columns")
    parser.add_argument("sparsity", type=float, help="Sparsity of matrix: (0.0,1.0] , note that higher density values will take longer to compute")
    parser.add_argument("--dtype", choices=list(dtypes.keys()),default="FP64", help="Data type")
    parser.add_argument("-s","--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate_random_sparse_mtx(args.fout,args.nrows,args.ncols, density=args.sparsity, dtype=dtypes[args.dtype], seed=args.seed)


if __name__ == "__main__":
    main()
