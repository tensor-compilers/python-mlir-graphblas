from mlir import execution_engine
from mlir import ir
from mlir import passmanager

# Import Kokkos pipeline
from torch_mlir_e2e_test.linalg_kokkos_backend import KokkosBackend

from .utils import MLIR_C_RUNNER_UTILS
import os

engine_cache = {}


def compile(module: ir.Module, 
            pipeline=None, *,
            opt_level=2,
            shared_libs=None,
            dump_mlir=True, 
            before_mlir_filename="dump_kgb.mlir",
            after_mlir_filename="lowered_dump_kgb.mlir",
            ws=os.getcwd(), 
            **kwargs
        ):
    if "index_instance" not in kwargs:
        kwargs["index_instance"] = 0
    if "num_instances" not in kwargs:
        kwargs["num_instances"] = 0
    if "options" not in kwargs:
        kwargs["options"] = ""
    backend = KokkosBackend.KokkosBackendLinalgOnTensorsBackend(dump_mlir=True, before_mlir_filename = before_mlir_filename, after_mlir_filename = after_mlir_filename, index_instance=kwargs["index_instance"], num_instances=kwargs["num_instances"])
    engine = backend.compile_sparse(module, options=kwargs["options"])
    return engine
