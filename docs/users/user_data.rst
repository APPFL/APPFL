User-defined dataset
===========================

User-defined dataset can be created simply by using the class defined below.
We expect in most cases this simple class would be sufficient. However, users can create more sophisticated dataset class for their own customization.

.. autoclass:: appfl.misc.data.Dataset
    :private-members: __len__, __getitem__
