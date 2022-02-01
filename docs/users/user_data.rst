How to define local dataset
===========================

User-defined dataset can be any class derived from ``torch.utils.data``.
We also provide a simple class ``appfl.misc.data.Dataset`` defined as follows:

.. literalinclude:: /../src/appfl/misc/data.py
    :lines: 1-14
    :language: python
    :caption: User-defined dataset class: src/appfl/misc/data.py


This allows users to easily create ``Dataset`` object with ``data_input: torch.FloatTensor`` and ``data_label: torch.Tensor``.
We expect in most cases this simple class would be sufficient. However, users can create more sophisticated dataset class for their own customization.
