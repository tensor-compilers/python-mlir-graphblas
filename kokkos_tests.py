from mlir_graphblas import tensor, operators, operations, io

import argparse
import numpy as np
from timeit import default_timer


def setup_tensor(mtxFile, matrix=True, sparse=True):
    if matrix:
        if sparse:
            return io.from_mtx(mtxFile, sparsity=["compressed","compressed"])
        else:
            # dense
            return io.from_mtx(mtxFile, sparsity=["dense","dense"])
    else:
        # vector
        if sparse:
            return io.from_mtx(mtxFile, sparsity=["compressed"])
        else:
            return io.from_mtx(mtxFile, sparsity=["dense"])


def vecadd(a, b, index_instance=0, num_instances=1, options=""):
    c = tensor.Vector.new(a.dtype, a.shape[0])
    operations.ewise_add(c, operators.BinaryOp.plus, a, b, index_instance=index_instance, num_instances=num_instances, options=options)
    return c


def vxv(a, b, index_instance=0, num_instances=1, options=""):
    c = tensor.Vector.new(a.dtype, a.shape[0])
    operations.ewise_mult(c, operators.BinaryOp.times, a, b, index_instance=index_instance, num_instances=num_instances, options=options)
    return c

def vxm(a, B, index_instance=0, num_instances=1, options=""):
    c = tensor.Vector.new(B.dtype, B.shape[1])
    operations.vxm(c,operators.Semiring.plus_times, a, B, index_instance=index_instance, num_instances=num_instances, options=options)
    return c

def mxv(A, b, index_instance=0, num_instances=1, options=""):
    c = tensor.Vector.new(A.dtype, A.shape[0])
    operations.mxv(c, operators.Semiring.plus_times, A, b, index_instance=index_instance, num_instances=num_instances, options=options)
    return c


def mxm(A, B, index_instance=0, num_instances=1, options=""):
    C = tensor.Matrix.new(A.dtype, A.shape[0], A.shape[1])
    operations.mxm(C, operators.Semiring.plus_times, A, B, index_instance=index_instance, num_instances=num_instances, options=options)
    return C


def matadd(A, B, index_instance=0, num_instances=1, options=""):
    C = tensor.Matrix.new(A.dtype, A.shape[0], A.shape[1])
    operations.ewise_add(C, operators.BinaryOp.plus, A, B, index_instance=index_instance, num_instances=num_instances, options=options)
    return C


def v_plus_v_plus_v(a, b, c, index_instance=0, num_instances=1,options=""):
    d = tensor.Vector.new(a.dtype, a.shape[0])
    operations.ewise_add(d, operators.BinaryOp.plus, b, c, index_instance=index_instance, num_instances=num_instances, options=options)
    operations.ewise_add(d, operators.BinaryOp.plus, a, d, index_instance=index_instance, num_instances=num_instances, options=options)
    return d


def m_plus_m_plus_m(A, B, C, index_instance=0, num_instances=1, options=""):
    D = tensor.Matrix.new(A.dtype, A.shape[0], A.shape[1])
    operations.ewise_add(D, operators.BinaryOp.plus, B, C, index_instance=index_instance, num_instances=num_instances, options=options)
    operations.ewise_add(D, operators.BinaryOp.plus, A, D, index_instance=index_instance, num_instances=num_instances, options=options)
    return D


def vxvxv(a, b, c, index_instance=0, num_instances=1, options=""):
    d = tensor.Vector.new(a.dtype, a.shape[0])
    operations.ewise_mult(d, operators.BinaryOp.times, b, c, index_instance=index_instance, num_instances=num_instances, options=options)
    operations.ewise_mult(d, operators.BinaryOp.times, a, d, index_instance=index_instance, num_instances=num_instances, options=options)
    return d


def mxmxm(A, B, C, index_instance=0, num_instances=1, options=""):
    D = tensor.Matrix.new(A.dtype, A.shape[0], A.shape[1])
    operations.mxm(D, operators.Semiring.plus_times, B, C, index_instance=index_instance, num_instances=num_instances, options=options)
    operations.mxm(D, operators.Semiring.plus_times, A, D, index_instance=index_instance, num_instances=num_instances, options=options)
    return D


def mxvxv(A, b, c, index_instance=0, num_instances=2, options=""):
    d = tensor.Vector.new(A.dtype, A.shape[0])
    operations.ewise_mult(d, operators.BinaryOp.times,b, c, index_instance=index_instance+1, num_instances=num_instances, options=options)
    operations.mxv(d, operators.Semiring.plus_times, A, d, index_instance=index_instance, num_instances=num_instances, options=options)
    return d


def main():
    operations_dict = {"v+v":(vecadd,'v','v'),
                       "vxv":(vxv,'v','v'),
                       "vxm":(vxm,'v','m'),
                       "mxv":(mxv,'m','v'),
                       "m+m":(matadd,'m','m'),
                       "mxm":(mxm,'m','m'),
                       "v+v+v":(v_plus_v_plus_v,'v','v','v'),
                       "vxvxv":(vxvxv,'v','v','v'),
                       "mxmxm":(mxmxm,'m','m','m'),
                       "mxvxv":(mxvxv,'m','v','v')}
    parser = argparse.ArgumentParser(prog="KokkosTest",description="For testing the Python-MLIR-GraphBLAS-Kokkos pipeline")
    parser.add_argument("mtxFile", help="Path to input matrix or vector in Matrix Market file format")
    parser.add_argument("operation", choices=list(operations_dict.keys()), help="Operation to perform")
    parser.add_argument("runtime", choices=["serial","openmp","gpu"], help="Computational runtime. Supports single-node serial, openmp, or gpu modes")
    parser.add_argument("-v","--verbose", action="store_true", help="Increase verbosity")
    parser.add_argument("--mtxout", help="Path to output Matrix Market file format. Note that this can be slow for large dense matrices")
    parser.add_argument("-s","--sparsity", choices=["sparse","dense"], default="sparse", help="Sparsity of the operands of either sparse or dense")  # TODO: mixed
    # parser.add_argument("--validate", action="store_true", help="Validate result against equivalent numpy operation (Note slow for large matrices)")
    args = parser.parse_args()

    # TODO: check if gpu is available and if kokkos is built with openmp/gpu
    if args.runtime != "serial":
        # gpu or openmp
        options="parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical"
    else:
        options = ""
    operation_func,*operand_types = operations_dict[args.operation]

    contains_matrix_operation = True if 'm' in operand_types else False
    is_dense = True if args.sparsity == "dense" else False
    operands = []

    timer = default_timer()
    if contains_matrix_operation:
        print("Reading as Matrix:",args.mtxFile)
        matrix = setup_tensor(mtxFile=args.mtxFile,matrix=True, sparse=not is_dense)
        print(f"Shape={matrix.shape}")
    else:
        print("Reading as Vector:",args.mtxFile)
        vector  = setup_tensor(mtxFile=args.mtxFile,matrix=False, sparse=not is_dense)
        print(f"Shape={vector.shape}")
    print(f"Finished reading mtx in {(default_timer() - timer):.4f} seconds")

    if contains_matrix_operation and 'v' in operand_types:
        print("Mixed tensor types, reading mtx as a vector too")
        timer2 = default_timer()
        vector  = setup_tensor(mtxFile=args.mtxFile,matrix=False, sparse=not is_dense)
        print(f"Shape={vector.shape}")
        print(f"Finished reading mtx in {(default_timer() - timer2):.4f} seconds")
    
    for operand in operand_types:
        if operand == 'm':
            operands.append(matrix)
        else:
            operands.append(vector)
    print("Finished preprocessing")

    print("Starting excecuting of", args.operation)
    timer2 = default_timer()
    out_tensor = operation_func(*operands, options=options)
    print(out_tensor.extract_tuples())
    print("Finished excecuting of operation")
    print(f"Time elapsed for execution (including compilation): {(default_timer() - timer2):.4f} seconds")
    if args.mtxout:
        print(f"Writing output to {args.mtxout}")
        timer2 = default_timer()
        io.to_mtx(args.mtxout, out_tensor)
        print(f"Finished writing tensor in {(default_timer() - timer2):.4f} seconds")
    print(f"Total time elapsed: {(default_timer() - timer):.4f} seconds")
    print("Done.")


if __name__ == "__main__":
    main()
