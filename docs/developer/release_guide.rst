Release Guide
=============

This page describes how to release the package. We basically follow the steps in `Packaging Python Projects <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.


TestPyPI
--------

We may want to test the distribution before uploading the archives to the PyPI server.
To this end, find the following lines in ``setup.py`` and change the project name to ``appfl-YOUR-NAME`` with a proper version.

.. code-block:: python

    setuptools.setup(
        name="appfl",
        version="0.0.1",
        ...
    )

.. note::

    Make sure to set a higher version if the same version exists in your TestPyPI server. You cannot upload the same version to the server.


Generating distribution archives
++++++++++++++++++++++++++++++++

Make sure that the ``dist`` directory at the root of the repository (where ``pyproject.toml`` is located) is empty or does not exist. Then, running the following command should generate a built distribution (``.whl``) and a source archive (``.tar.gz``) in the ``dist`` directory.

.. code-block:: shell

    python -m build


Uploading the distribution archives
+++++++++++++++++++++++++++++++++++

Upload the distribution archives to the test server with the following command. This requires an account on `TestPyPI <https://test.pypi.org>`_.

.. code-block:: shell

    twine upload --repository testpypi dist/*


Installing the new test package
+++++++++++++++++++++++++++++++

Once the distribution archives are uploaded, anyone can install the new test package with the following command:

.. code-block:: shell

    pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple "appfl-YOUR-NAME[dev,examples,analytics]"


Release
-------

If the testing above is successful, we are ready to release the new package to the PyPI server.
The key steps are:

- Change the project name back and set the version properly.
- Generate the distribution archives as above.
- Use ``twine upload dist/*`` to upload the distribution archives.
