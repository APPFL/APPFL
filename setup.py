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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="appfl",
    version="1.0.0",
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
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "grpcio",
        "grpcio-tools",
        "omegaconf",
        "globus-sdk",
        "mpi4py",
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
        "torchvision",
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
        ],
        "examples": [
            "opencv-python",
            "pandas",
            "pyarrow",
            "fastparquet",
            "tqdm",
            "jupyter",
            "wget",
        ],
    },
    entry_points={
        "console_scripts": [
            "appfl-auth=appfl.login_manager.globus.cli:auth",
            "appfl-install-compressor=appfl.compressor.install:install_compressor"
        ],
    },
)
