# Documentation

> Good documentation is very important.

We build the documentation with [Sphinx](https://www.sphinx-doc.org), which also supports Markdown,
so both reStructuredText and Markdown files can be used.

## Required packages

Install the required packages with:

```shell
pip install "appfl[dev]" --upgrade
```

We use `nbsphinx` to include Jupyter notebooks in the docs, which requires `pandoc`. You may need to install it manually with conda:

```shell
conda install -c conda-forge pandoc
```

## Build document

Build the documentation with:

```shell
cd APPFL/docs
make html
```

You can also preview the docs in VS Code.
