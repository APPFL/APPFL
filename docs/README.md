# How to document

We can do the documentation based on https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
Markdown syntax can be supported by using MyST in the readthedocs manual.

The following python packages are required to build documents using Sphinx:
```console
$ pip install sphinx sphinx-rtd-theme myst-parser
```

For now, you can build the document on local.

```
make html
```

Or, `vscode` can preview the docs.
