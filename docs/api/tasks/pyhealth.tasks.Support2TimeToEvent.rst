pyhealth.tasks.Support2TimeToEvent
==================================

Use :class:`~pyhealth.datasets.Support2Dataset` with the ``support2`` table to
predict all-cause mortality from study entry. Each eligible patient contributes
one sample with a float32 follow-up duration in days from ``d.time`` and an event
indicator from ``death`` (1 for observed death, 0 for right censoring).

Pass a :class:`~pyhealth.datasets.PatientSplit` to ``set_task`` so the age
imputation and scaling statistics and the sex vocabulary are fitted on training
patients. The processed features contain standardized age, an age-missing
indicator, one column per training sex category, and separate columns for missing
and unseen sex values.

The repository's ``examples/survival_mdn_support2.ipynb`` shows how to prepare the
HBiostat archive, select a checkpoint by validation loss, and export held-out
predictions on a day grid with :class:`~pyhealth.models.SurvivalMDN``.

.. autoclass:: pyhealth.tasks.Support2TimeToEvent
    :show-inheritance:
