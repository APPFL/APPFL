import setuptools

# Build Author list
authors = {
    "Kibaek Kim": "kimk@anl.gov",
    "Minseok Ryu": "mryu@anl.gov",
    "Youngdae Kim": "youngdae@anl.gov",
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
    version="0.2.0",
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
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "grpcio",
        "grpcio-tools",
        "omegaconf",
    ],
    extras_require={
        "dev": [
            "sphinx",
            "sphinx_rtd_theme",
            "myst-parser",
            "nbsphinx",
            "twine",
            "build",
            "black",
            "pytest",
            "pytest-mpi",
            "torchvision",
            "mpi4py",
        ],
        "analytics": [
            "jupyter",
            "tensorboard",
            "matplotlib",
        ],
        "mpi": ["mpi4py"],
        "examples": [
            "opencv-python",
            "torchvision",
        ],
    },
)
