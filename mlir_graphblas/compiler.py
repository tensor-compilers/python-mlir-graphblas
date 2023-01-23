from mlir import execution_engine
from mlir import ir
from mlir import passmanager

from .utils import LIBMLIR_C_RUNNER_UTILS


engine_cache = {}


def compile(module: ir.Module, pipeline=None, *, opt_level=2, shared_libs=None):
    if pipeline is None:
        pipeline = 'builtin.module(sparse-compiler{reassociate-fp-reductions=1 enable-index-optimizations=1})'
    if shared_libs is None:
        shared_libs = [LIBMLIR_C_RUNNER_UTILS]
    # print(module)
    passmanager.PassManager.parse(pipeline).run(module)
    return execution_engine.ExecutionEngine(
        module, opt_level=opt_level, shared_libs=shared_libs)
