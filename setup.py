from setuptools import find_packages, setup
from mlir_graphblas import __version__


extras_require = {
    "test": ["pytest"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

with open("README.md") as f:
    long_description = f.read()

setup(
    name="python-mlir-graphblas",
    version=__version__,
    description=(
        "GraphBLAS Implementation written using MLIR Python Bindings"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jim Kitchen",
    author_email="jim22k@gmail.com",
    url="https://github.com/python-graphblas/python-mlir-graphblas",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["mlir-python-bindings"],
    extras_require=extras_require,
    include_package_data=True,
    license="Apache License 2.0",
    keywords=["graphblas", "graph", "sparse", "matrix", "mlir", "sparse_tensor"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)