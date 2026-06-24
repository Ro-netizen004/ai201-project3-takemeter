# TakeMeter Planning — IMDb Movie Review Community

## Community

**Community:** IMDb user reviews (public movie review discourse).

**Why this community:** IMDb is a large, text-heavy online space where people constantly share opinions about films. Review quality and tone vary widely — some reviews are detailed critical essays, others are brief praise or venting. The positive vs. negative sentiment distinction is meaningful to readers deciding whether a film is worth watching, and it is easy to apply consistently at scale.

**Why it fits classification:** Reviews are short-to-medium text, publicly available, and carry a natural binary label (sentiment) that reflects how the community expresses approval or disapproval.

---

## Labels

Two mutually exclusive labels:

| Label | Definition |
|-------|------------|
| `negative` | The review expresses criticism, disappointment, or dislike of the film — the reviewer would not recommend it or found it unsatisfying. |
| `positive` | The review expresses praise, enjoyment, or approval — the reviewer liked the film or would recommend it. |

### Examples per label

**negative**
1. *"A complete waste of two hours. The plot made no sense and the acting was wooden throughout."*
2. *"I walked out halfway through. Boring, predictable, and poorly edited."*

**positive**
1. *"One of the best films I've seen this year. The cinematography is stunning and the performances are heartfelt."*
2. *"Loved every minute. Funny, moving, and tightly paced — highly recommend."*

### Hard edge cases

**Ambiguous case:** A review that mixes praise and criticism but lands on one overall verdict.

*Example:* "The first half was brilliant but the ending ruined it. Still worth watching for the performances alone."

**Decision rule:** Label by the reviewer's **overall recommendation**. If they would still tell someone to watch it (or express net satisfaction), label `positive`. If the dominant takeaway is disappointment or "don't bother," label `negative`. Mixed reviews that end with a clear thumbs-up → `positive`; mixed reviews that end with regret → `negative`.

**During annotation:** This dataset uses the official IMDb labels from HuggingFace (`datasets.load_dataset("stanfordnlp/imdb")`), which were assigned by human annotators using similar overall-sentiment criteria.

---

## Data Collection Plan

- **Source:** HuggingFace `stanfordnlp/imdb` dataset (50k train + 25k test reviews, Apache 2.0).
- **Target size:** 300 examples (150 per label) for a balanced set above the 200 minimum.
- **Collection:** `scripts/prepare_imdb_dataset.py` samples stratified examples and writes `data/imdb_labeled.csv` with columns `text`, `label`, `notes`.
- **If imbalanced:** Script enforces 50/50 split; no label may exceed 70%.
- **Split:** Notebook handles 70% / 15% / 15% train/val/test automatically.

---

## Evaluation Metrics

| Metric | Why |
|--------|-----|
| **Accuracy** | Binary task with balanced classes — overall correctness is interpretable. |
| **Per-class precision, recall, F1** | Shows whether the model favors one sentiment; F1 is the primary per-class summary. |
| **Confusion matrix** | Reveals false positives (negative called positive) vs. false negatives. |

Accuracy alone is insufficient: a model could score 50% on a balanced set by guessing, so per-class F1 and the confusion matrix matter more for diagnosing failures.

---

## Definition of Success

| Criterion | Threshold |
|-----------|-----------|
| Fine-tuned model beats zero-shot Groq baseline on test accuracy | ≥ 5 percentage points |
| Per-class F1 (fine-tuned) | ≥ 0.75 for both labels |
| Test accuracy (fine-tuned) | ≥ 0.80 |

"Good enough for deployment" = a moderation or recommendation sidebar could flag review tone with ≥80% accuracy and no single class F1 below 0.70.

---

## AI Tool Plan

### Label stress-testing
Before committing to IMDb, I asked an AI to generate borderline reviews (mixed praise/criticism). Several could not be labeled cleanly without the overall-recommendation rule above — that confirmed binary sentiment with an explicit tie-break rule is workable.

### Annotation assistance
**Used:** The HuggingFace IMDb labels are pre-annotated by the dataset authors. I did not manually re-label 300 rows; I sampled from the official train split and verified label distribution. Disclosure in README AI usage section.

### Failure analysis
After training, I will paste misclassified test examples into an AI tool and ask for patterns (sarcasm, short reviews, neutral tone). I will verify each pattern against the confusion matrix and raw examples before writing the evaluation report.

---

## Difficult Labeling Decisions (from dataset review)

1. **Sarcastic praise:** *"Oh great, another sequel nobody asked for. Truly groundbreaking."* — Sounds positive literally but is `negative`. IMDb label: `negative` ✓
2. **Qualified positive:** *"Flawed but I enjoyed it."* — Borderline; net enjoyment → `positive`. IMDb label: `positive` ✓
3. **Technical criticism with mild praise:** *"Great visuals, terrible script."* — Without a clear overall verdict, could go either way; IMDb human labels vary here — noted as a known failure mode for the model.
