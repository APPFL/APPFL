Subgroup Calibration with CADRE
===============================

This module allows multiple hospitals to assess a model's calibration without
sending all of their patient-level predictions to a central location. The
module uses the CADRE structure developed by APPFL.

Each hospital organizes its predictions into the same probability bins. For
instance, if there are ten bins, predictions that range between 0% and 10% are
categorized into the first bin, predictions that range between 10% and 20% are
categorized into the second bin, and so on.

The hospitals do this separately for each demographic group. The hospital will
send the following statistics to the server for each demographic group and
probability bin: the number of patients, the total of the probabilities of each
patient, the number of patients who actually experienced an outcome of interest,
and the total squared error of the model's predictions.

The server collects the summaries from each hospital and calculates two metrics
to evaluate the model's performance. The first metric is the expected
calibration error (ECE) of the model, which measures how well the model's
predictions match the actual outcomes of the patients. The second metric is the
Brier score of the model, which measures the overall accuracy of the model's
predictions.

Because all hospitals use the same probability bins, these summaries contain
enough information to calculate both metrics. The results are the same as if the
same retained patients were combined into a single dataset using the released
bins, except for the small differences introduced by floating-point rounding.

Run the example
---------------

This example includes four synthetic hospitals with different sample sizes,
outcome rates, demographic group sizes, and calibration patterns. These examples
demonstrate how the module functions without training a model.

To run the example from the APPFL repository root after installing APPFL, use
these commands.

.. code-block:: bash

   python -X utf8 examples/data_readiness/subgroup_calibration/run.py
   python -X utf8 examples/data_readiness/subgroup_calibration/run.py --network

The first command runs serially. The second command uses the same reports as the
first command but sends them over a local gRPC server. The reports are saved to
``output/subgroup_calibration`` in both cases.

Configure the module
--------------------

.. code-block:: yaml

   client_configs:
     data_readiness_configs:
       generate_dr_report: true
       output_dirname: ./output/subgroup_calibration
       dr_metrics:
         cadremodule_configs:
           cadremodule_path: ./examples/data_readiness/subgroup_calibration/cadre_module.py
           cadremodule_name: SubgroupCalibrationCADREModule
           remedy_action: false
           cadremodule_kwargs:
             n_bins: 10
             release_n_bins: 5
             min_cell_count: 5
             min_outcome_count: 0

Each hospital's dataset must contain three one-dimensional arrays that are
aligned with one another:

* ``predictions``: predicted probabilities between 0 and 1
* ``outcomes``: actual binary outcomes, either 0 or 1
* ``groups``: the demographic group label for each patient

The predictions should be made using held-out data. The ``server.yaml`` file in
this repository is an example of how to configure the calibration evaluation.
The network example runs on one machine with unauthenticated loopback gRPC and
synthetic data.

Bins and suppression
--------------------

Some subgroups and probability ranges may only contain one or two patients. The
code can combine these small ranges with their neighbors before reporting the
totals for those ranges. Also, the code can drop any ranges that do not contain
a sufficient number of patients.

``n_bins`` controls how many equal-width probability bins are created. For
example, with ``n_bins: 10``, the bins are 0-0.1, 0.1-0.2, and so on.

``release_n_bins`` controls how many bins are released after neighboring bins
are combined. It must evenly divide ``n_bins``. All hospitals must use the same
settings, chosen before evaluating the data.

This example takes the ten ranges and combines them into five. It also removes
any local subgroup/bin cell that contains fewer than five patients. These
settings are applied to all hospitals. The resulting ECE and Brier scores
reflect the data that was not dropped.

Before sending a report, each client merges its bins locally. Cells with fewer
than ``min_cell_count`` patients are left out. ``min_outcome_count`` can also
require a minimum number of both positive and negative outcomes in a cell.
Setting it to zero turns this check off.

Reports contain only the cells that pass these checks and the configuration
settings.

The ``--min-outcomes 1`` option requires at least one positive and one negative
outcome in each reported cell. This can result in the removal of patients when
outcomes are rare. Using wider ranges can hide differences in calibration
between the models. While these options reduce the amount of data that is
disclosed about individual patients, they do not provide any formal guarantee
of patient privacy.

The server checks each hospital's report before combining the totals. If no
cells remain for a subgroup after suppression, the server cannot calculate a
score for that subgroup. The results clearly state that the scores only include
patients from cells that were retained.

For a subgroup with ``N`` retained patients, ECE is calculated as:

``sum_b(abs(sum_p_b - sum_y_b)) / N``

Brier score is calculated as:

``sum_b(sum_sq_err_b) / N``

These formulas only require the counts and sums from each bin. Therefore,
combining the hospitals' summaries gives the same result as pooling the same
retained patient rows, apart from small differences caused by computer
rounding.
