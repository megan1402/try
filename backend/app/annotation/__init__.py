"""LLM-assisted annotation.

Phase 1 of the AI-assisted labelling plan: evaluate whether an LLM can reproduce
the existing human annotations before generating any new ones.

* :mod:`app.annotation.schema` — the Pydantic contract every response must meet.
* :mod:`app.annotation.prompt` — prompt construction from the human rubric.
* :mod:`app.annotation.llm` — the annotator, with an injectable transport.
* :mod:`app.annotation.batch` — checkpointed batch runs with provenance.
* :mod:`app.annotation.compare` — agreement with the human labels.

**AI labels are never ground truth.** Every stored record carries
``human_verified: false`` until a human overrides it, and the comparison module
treats the human labels as the reference, never the reverse.
"""
