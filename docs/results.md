# Results

Every measurement in the project, in the order it was made. Each entry states
what was measured, what it showed, and what it does **not** show.

The short version: **no extractor works well enough yet, so the event study has
not been meaningfully tested.**

| # | Measurement | Result |
|---|---|---|
| 1 | Rule detector, precision | 0.055 |
| 2 | Rule detector, recall | 0.000 |
| 3 | Event study, 41 issuers | no abnormal returns |
| 4 | Zero-shot NLI | precision 0.156, recall 0.500 |
| 5 | DeepSeek LLM, old dev set | precision 0.153, recall 0.917 |
| 6 | **DeepSeek LLM, new dev partition** | **precision 0.767, recall 0.852** |

---

## 1. Rule detector — precision 0.055

**Measured:** 149 stratified detected events, labelled by hand.

| class | precision | 95% CI |
|---|---|---|
| acquisition | 0.076 | 0.035 – 0.156 |
| legal_or_regulatory | 0.105 | 0.029 – 0.314 |
| restructuring | 0.000 | 0.000 – 0.176 |
| divestiture | 0.000 | 0.000 – 0.278 |
| five others | 0.000 | uninterpretable (n < 10) |

**Pooled: 0.055 [0.028, 0.104].** Of 149 detected events, 8 were real.

**Why:** the rules match vocabulary, not events. "Acquisitions, net of cash
acquired and divestitures (516)" is a cash-flow row. "Shares acquired under the
repurchase programme" is a buyback.

**Does not show:** that filings do not move prices. It shows the detector finds
almost no real events.

---

## 2. Rule detector — recall 0.000

**Measured:** 38 event-bearing passages, sampled independently of the detector.

| class | found | recall |
|---|---|---|
| acquisition | 0 of 1 | 0.000 |
| management_change | 0 of 1 | 0.000 |

It also fired on 2 of 36 event-free passages.

**Why:** the misses are semantic, not lexical. *"Following Mr. Huang's
resignation, the Board appointed..."* puts the verb before the role; the pattern
requires the reverse.

**Does not show:** a precise recall estimate. With 2 real events the interval is
0.000–0.793.

---

## 3. Event study — no abnormal returns

**Measured:** 41 issuers, 516 filings, 297 events, 118 distinct anchor dates.

| class | n_dates | median 1d | p (1d) |
|---|---|---|---|
| acquisition | 105 | 0.000 | 0.98 |
| restructuring | 39 | −0.004 | 0.79 |
| legal_or_regulatory | 22 | +0.001 | 0.98 |
| divestiture | 12 | +0.009 | 0.59 |

Every p-value between 0.59 and 0.99. Event medians sit at the 31st–70th
percentile of ordinary trading days.

**Why this is not a finding about the market:** the events measured were mostly
not events. Precision was 0.055, so the study tested noise.

**Does not show:** that filing events do not predict returns. That remains
untested.

---

## 4. Zero-shot NLI — precision 0.156, recall 0.500

**Measured:** `cross-encoder/nli-distilroberta-base`, threshold swept 0.0–0.9.

| threshold | predicted | precision | recall |
|---|---|---|---|
| 0.0 | 146 | 0.055 | 0.800 |
| 0.5 | 45 | **0.156** | 0.700 |
| 0.7 | 8 | 0.125 | 0.100 |

**Why it fails:** precision peaks at 0.156 and then *falls*. If the scores
carried signal, confidence would correlate with correctness. It does not.

**Does not show:** a verdict on LLMs. This is a 330 MB NLI model, not a
frontier model.

---

## 5. DeepSeek LLM — precision 0.153, recall 0.917

**Measured:** `deepseek-chat`, all 187 human-labelled passages, blind to labels
and detector output.

| metric | value | 95% CI |
|---|---|---|
| precision | 0.153 | 0.088 – 0.253 |
| recall | 0.917 | 0.646 – 0.985 |
| specificity | 0.645 | 0.571 – 0.713 |
| accuracy | 0.663 | 0.592 – 0.727 |

**The LLM finds events the rules miss entirely** — 11 of 12, versus 0. But it
over-fires by roughly 6×.

**Accuracy 0.663 is below the always-`none` baseline of 0.935.** The dataset is
93% `none`, so accuracy is not the metric to quote.

**Why it fails:** three rubric gaps, not model failure — undated acquisitions,
announced-but-unclosed transactions, and section headings. The model applies the
rubric correctly where the rubric is silent.

**Does not show:** a test-set result. These 187 labels are a development set the
rubric was tuned on.

---

## 6. DeepSeek LLM, new development partition — precision 0.767, recall 0.852

**Measured:** the same model and prompt, but on 80 freshly labelled snippets from
18 companies, after three rubric rules were added (undated events, unclosed
transactions, headings).

| metric | value | 95% CI |
|---|---|---|
| precision | **0.767** | 0.591 – 0.882 |
| recall | **0.852** | 0.675 – 0.941 |
| specificity | 0.868 | 0.752 – 0.935 |
| accuracy | 0.863 | 0.770 – 0.921 |

Per class:

| class | support | predicted | precision | recall |
|---|---|---|---|---|
| acquisition | 16 | 18 | 0.667 | 0.750 |
| legal_or_regulatory | 8 | 9 | 0.889 | 1.000 |
| divestiture | 2 | 2 | 1.000 | 1.000 |
| restructuring | 1 | 1 | 1.000 | 1.000 |
| none | 53 | 50 | 0.920 | 0.868 |

**This is a five-fold improvement in precision** over the same model on the old
set (0.153 → 0.767), with recall essentially unchanged (0.917 → 0.852).

**Why:** the three rubric rules closed the gaps the model was falling into. The
model was never ignoring the rubric — it was applying it where the rubric was
silent.

**The remaining errors are the opposite of before.** The model now *misses*
headings it should have caught:

| id | passage | human | model |
|---|---|---|---|
| ev0065 | "Acquisition of VMware, Inc." | acquisition | none |
| ev0088 | "Acquisition of Symantec Corporation's Enterprise Security Business" | acquisition | none |

Those are headings, which the rubric says are `none` — so the model is following
the rule and the human is not. That is a labelling inconsistency, not a model
error.

**Does not show:** a test-set result. These 80 snippets are new data, but the
rubric was refined while labelling them, so they are development data.

---

## What the results mean together

| approach | precision | recall | verdict |
|---|---|---|---|
| Rule baseline | 0.055 | 0.000 | does not detect |
| Zero-shot NLI | 0.156 | 0.500 | fires on everything |
| DeepSeek, old dev set | 0.153 | 0.917 | finds events, over-fires 6× |
| **DeepSeek, new dev partition** | **0.767** | **0.852** | **usable** |

**Extraction is no longer the bottleneck.** With the revised rubric the LLM
reaches 0.767 precision at 0.852 recall on 80 snippets — good enough that the
event study could be re-run on its output.

The caveat is that this is development data. The test partition (70 snippets
from 12 companies) is still unlabelled, and it is the only set that can confirm
the result generalises.

## What has not been measured

- **The test partition.** 70 snippets, unlabelled. This is the number that
  matters.
- **Whether the result holds on a wider corpus.** 80 snippets from 18 large-cap
  issuers.
- **Whether the event study finds anything with real events.** Blocked on the
  above.
- **Whether real filing events predict returns.** Blocked on extraction.
