"""Evaluation: turning "it looks right" into a number.

* :mod:`app.evaluation.gold` — the labelled snippet set and its file format.
* :mod:`app.evaluation.review` — the terminal tool used to label it.
* :mod:`app.evaluation.report` — precision / recall / F1 for the rule baseline.

Nothing here calls a model or the network: the same harness will score a Hugging
Face classifier later, which is the point of building it before the model exists.
"""
