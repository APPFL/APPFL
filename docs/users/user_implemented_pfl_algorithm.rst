Personalization algorithms
==========================

Various methods for privacy-preserving personalization @ FL server and client
-----------------------------------------------------------------------------
APPFL supports implementation of personalization schemes, wherein certain layers are marked as personalized, and remain local to the client. Personalized layers are updated locally at each client, and do not participate in the FL process, thereby ensuring data localization and privacy.

- |PersonalizedClientOptim|_: personalization layers (loss captures full dataset)
- |PersonalizedClientStepOptim|_: personalization layers (loss captures stochastic minibatch)

Both methods support `differential privacy <https://arxiv.org/abs/2312.00036>`_ when the ``cfg.fed.args.use_dp`` flag is set for APPFL's configuration object in-code.

.. |PersonalizedClientOptim| replace:: ``PersonalizedClientOptim``
.. _PersonalizedClientOptim: https://arxiv.org/abs/2309.13194
.. |PersonalizedClientStepOptim| replace:: ``PersonalizedClientStepOptim``
.. _PersonalizedClientStepOptim: https://arxiv.org/abs/2309.13194

Implementation of `differential privacy <https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1>`_ requires that client-to-server updates are clipped to a constant upper bound. This can be achieved by setting ``cfg.fed.args.clip_value`` and ``cfg.fed.args.clip_norm``, and specifying the norm-type by setting ``cfg.fed.args.clip_norm``. 
Currently, APPFL supports `Laplace mechanism <https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1>`_ for differential privacy, which requires usage of 1-norm for privacy guarantees to be meaningful.

The final privacy budget is given as ``cfg.fed.args.clip_value`` / ``cfg.fed.args.clip_norm``.

Please see ``How to set configuration`` for more details about setting configuration parameters. Theoretical details are included in the linked papers.