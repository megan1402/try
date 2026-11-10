"""Optional model layer.

Everything here needs the *optional* ML extras (``requirements-ml.txt``) and is
therefore imported lazily and excluded from CI. The evaluation harness does not
care: any callable ``str -> label`` can be scored.

* :mod:`app.models.zero_shot` — an NLI model used as an untrained classifier.
* :mod:`app.models.compare` — rules versus model, side by side.
"""
