# Documentation

> Good documentation is very important.

We make and update the documentation based on [Sphinx](https://www.sphinx-doc.org) with some support of Markdown. 
So we can use both reStructuredText and Markdown files.
We use [Sphinx Book Theme](https://sphinx-book-theme.readthedocs.io/en/latest/index.html).

## Installation

To this end, we need to install the following packages:

```shell
pip install sphinx sphinx-book-theme myst-parser
```

We use `nbsphinx` to add Jupyter notebooks to the doc, which requires `pandoc` package. This pakcage may need to be manually installed from conda.

```shell
conda install -c conda-forge pandoc
```

## Build

The build command for documentation is:

```shell
cd appfl/docs
make html
```

Note that `vscode` can preview the docs.
