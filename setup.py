import sys
import setuptools

# Build Author list
authors = {
    "Kibaek Kim": "kimk@anl.gov",
    "Minseok Ryu": "minseok.ryu@asu.edu",
    "Youngdae Kim": "youngdae@anl.gov",
    "Hieu Hoang": "thoang@anl.gov",
    "Zachary Ross": "zross@anl.gov",
    "Zilinghan Li": "zilinghan.li@anl.gov",
    "Sang-il Yim": "yim@anl.gov",
    "Shourya Bose": "shbose@ucsc.edu",
    "Shilan He": "shilanh2@illinois.edu",
    "Grant Wilkins": "gfw27@cam.ac.uk",
    "Ravi Madduri": "madduri@anl.gov",
}
AUTHOR = ""
for i, (k, v) in enumerate(authors.items()):
    if i > 0:
        AUTHOR += ", "
    AUTHOR += f"{k} <{v}>"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

if sys.version_info >= (3, 9):
    numpy_version = "numpy==1.26.4"
else:
    numpy_version = "numpy"  # Default numpy version for Python < 3.9

setuptools.setup(
    name="appfl",
    version="1.6.1",
    author=AUTHOR,
    description="An open-source package for privacy-preserving federated learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/APPFL/APPFL",
    project_urls={
        "Bug Tracker": "https://github.com/APPFL/APPFL/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "": ["*.sh", "*.crt", "*.key"],
    },
    python_requires=">=3.8",
    install_requires=[
        numpy_version,
        "torch",
        "grpcio",
        "grpcio-tools",
        "omegaconf",
        "globus-sdk",
        "globus-compute-sdk",
        "globus-compute-endpoint",
        "boto3",
        "botocore",
        "lz4",
        "zfpy",
        "zstd",
        "blosc",
        "python-xz",
        "matplotlib",
        "seaborn",
        "piq",
        "scikit-learn",
        "wandb",
        "colorama",
        "proxystore[all]",
        "ray[default]",
        "pandas",
    ],
    extras_require={
        "dev": [
            "sphinx",
            "sphinx_rtd_theme",
            "pydata-sphinx-theme",
            "sphinx-design",
            "sphinx-copybutton",
            "sphinx-contributors",
            "myst-parser",
            "nbsphinx",
            "twine",
            "build",
            "black",
            "pytest",
            "pytest-mpi",
            "furo",
        ],
        "examples": [
            "opencv-python",
            "pyarrow",
            "fastparquet",
            "tqdm",
            "jupyter",
            "wget",
            "torchvision",
            "tensorboard",
        ],
        "monai": ["monai[all]==1.2.0"],
        "mpi": ["mpi4py"],
    },
    entry_points={
        "console_scripts": [
            "appfl-auth=appfl.login_manager.globus.cli:auth",
            "appfl-install-compressor=appfl.compressor.install:install_compressor",
            "appfl-setup-ssl=appfl.comm.grpc.setup_ssl:setup_ssl",
        ],
    },
)
