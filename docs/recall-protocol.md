# Recall protocol

How to label the passage sample, and why it is the missing half of the
evaluation.

## Why this exists

Every number measured so far is **precision-only**, because the samples were
drawn from events the detector reported. That cannot measure recall: the events
it missed are absent by construction.

The measured precision is **0.055** (see `docs/precision-result.md`). That tells
you the detector is usually wrong when it fires. It does **not** tell you whether
it also misses most real events — a detector that reports one event per filing
and is right every time has precision 1.0 and recall near zero.

This sample fixes that by drawing **passages**, not detections.

## What is sampled

Paragraphs of filing prose, chosen at random from every document in the corpus,
with no reference to what the detector thinks. Three per document by default,
seeded (`20260121`), so the sample is reproducible.

The detector's answers are stored in a **sidecar file**
(`passages_recall.detected.json`) and are not shown while labelling. If they
were, the recall estimate would be circular.

## The rubric

Same as `docs/precision-protocol.md`, with one addition: **label the passage,
not the document.** A passage that mentions an acquisition in passing is an
`acquisition` passage; a passage about accounting policy is `none`.

### Label it as an event when

- The passage **reports** the event: *"We completed the acquisition of Acme."*
- The passage **describes** an in-period event: *"Our acquisition of Semrush in
  April 2026 enhances our offerings."*
- The passage **states** a change in outlook, a management change, or a legal
  outcome.

### Label it `none` when

- **Boilerplate**: "Our acquisition strategy focuses on tuck-in deals."
- **Accounting**: "Property and equipment acquired was $1.2 billion."
- **Hypothetical**: "An acquisition could harm our business."
- **Historical**: "In 2019 we acquired Mellanox." — prior period.
- **Cross-reference**: "See Item 1A for litigation risk."
- **Ordinary results**: "Revenue grew 12% year over year."

### Label it `skip` when

The passage is truncated, or the event is real but the class is genuinely
arguable. Skipped passages are excluded from every metric.

## How to label

```bash
make review-passages     # one keystroke per passage
make recall              # recall per class + confusion matrix
```

## Reading the output

```
event_type   true_positives  false_negatives  recall  ci_low  ci_high
acquisition              12               31   0.279   0.166    0.430
```

- **`false_negatives`** is the number that matters. These are real events the
  detector did not report, and they are invisible to any sample drawn from
  detections.
- **`recall`** is `tp / (tp + fn)`.
- **`ci_low`/`ci_high`** is a 95% Wilson interval.

The confusion matrix shows the full picture:

```
labelled   detected        count
acquisition acquisition      12     <- true positives
acquisition none             31     <- false negatives (the point of this exercise)
none        acquisition      44     <- false positives
none        none            180     <- true negatives
```

## What a good result looks like

- **High recall, low precision** (e.g. 0.8 / 0.1): the detector finds most real
  events but also fires on everything. Fixable by tightening.
- **Low recall, low precision** (e.g. 0.2 / 0.05): the detector is not working.
  Needs a different approach, not tuning.
- **Low recall, high precision**: conservative but incomplete. The event study
  would be measuring a biased subset.

Given precision is 0.055, the interesting question is whether recall is also
low. If both are low, keyword matching is simply the wrong tool.

## Known limitations

- **Passage-level, not document-level.** A document with one real event among
  fifty paragraphs contributes one positive passage. Recall measured this way is
  about *passages*, not *documents*.
- **One annotator.** No inter-annotator agreement, so the noise floor is
  unknown.
- **Sampling error.** With ~300 passages the intervals are wide; a difference of
  0.1 between two classes may not be real.
- **The detector is applied to the passage alone**, not the whole document. That
  is the honest comparison — a reviewer sees only the passage — but it means
  section-based filters cannot help, so recall here is a lower bound on what the
  detector achieves on full documents.
