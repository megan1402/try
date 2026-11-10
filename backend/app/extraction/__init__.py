"""Turn raw filing documents into structured text.

Ingestion (:mod:`app.ingestion`) only *retrieves* documents; this package is
where their contents start to mean something:

* :mod:`app.extraction.html` — HTML/XHTML into plain text, preserving block
  structure.
* :mod:`app.extraction.sections` — plain text into Part/Item sections.

Event extraction (10-K/10-Q/8-K -> structured events) builds on top of these
two and is the next step.
"""
