# TakeMeter — IMDb Review Sentiment Classifier

**AI201 · Project 3**

**Demo video:** [https://streamable.com/6juaxa](https://streamable.com/6juaxa)

TakeMeter classifies IMDb movie reviews as **negative** or **positive** based on overall sentiment. This repo contains the planning document, labeled dataset, Colab training notebook, and evaluation results comparing a fine-tuned DistilBERT model to a zero-shot Groq baseline.

---

## Community Choice

**Community:** IMDb user reviews (public movie review discourse).

IMDb is a text-heavy online community where people constantly share opinions about films. Review tone varies from detailed criticism to brief praise. The positive vs. negative distinction matters to readers deciding whether a film is worth watching, and the labels are grounded in how reviewers express approval or disapproval.

---

## Label Taxonomy

| Label | Definition |
|-------|------------|
| `negative` | The review expresses criticism, disappointment, or dislike — the reviewer would not recommend the film or found it unsatisfying. |
| `positive` | The review expresses praise, enjoyment, or approval — the reviewer liked the film or would recommend it. |

### Examples

**negative**
- *"A complete waste of two hours. The plot made no sense and the acting was wooden throughout."*
- *"I walked out halfway through. Boring, predictable, and poorly edited."*

**positive**
- *"One of the best films I've seen this year. The cinematography is stunning and the performances are heartfelt."*
- *"Loved every minute. Funny, moving, and tightly paced — highly recommend."*

---

## Data Collection

| Item | Detail |
|------|--------|
| **Source** | HuggingFace `stanfordnlp/imdb` (official Stanford sentiment dataset) |
| **Size** | 300 reviews (150 negative, 150 positive) |
| **Split** | 70% train / 15% validation / 15% test (45 test examples) — handled in notebook |
| **Labeling** | Labels come from the dataset's human annotators; sampled with `scripts/prepare_imdb_dataset.py` |
| **CSV** | `data/imdb_labeled.csv` (columns: `text`, `label`, `notes`) |

### Label distribution

| Label | Count | % |
|-------|------:|--:|
| negative | 150 | 50% |
| positive | 150 | 50% |
| **Total** | **300** | **100%** |

### Difficult-to-label examples

1. **Mixed verdict ending positive:** *"The first half was brilliant but the ending ruined it. Still worth watching for the performances alone."* → Labeled `positive` (net recommendation wins).
2. **Sarcastic praise:** *"Oh great, another sequel nobody asked for. Truly groundbreaking."* → Labeled `negative` (literal positive words, negative intent).
3. **Split praise/criticism:** *"Great visuals, terrible script."* → Borderline; labeled by overall verdict in source data — a known failure mode for both models.

---

## Fine-Tuning Approach

| Setting | Value |
|---------|-------|
| **Base model** | `distilbert-base-uncased` (HuggingFace) |
| **Training** | Google Colab, T4 GPU, `transformers` Trainer API |
| **Epochs** | 3 |
| **Learning rate** | 2e-5 (default) |
| **Batch size** | 16 |
| **Max sequence length** | 256 tokens |

**Hyperparameter decision:** Kept the default learning rate of **2e-5** because it is the standard starting point for BERT-family fine-tuning on small text classification sets. With only ~210 training examples, a higher rate risked instability; a lower rate would slow convergence without clear benefit on this task.

---

## Baseline

| Item | Detail |
|------|--------|
| **Model** | Groq `llama-3.3-70b-versatile` |
| **Approach** | Zero-shot — no task-specific training |
| **Prompt** | System prompt with label definitions, one example per label, and instruction to output only the label name (`negative` or `positive`) |
| **Temperature** | 0 |
| **Evaluation** | Same locked 45-example test set as the fine-tuned model |

---

## Evaluation Report

### Overall accuracy (test set, n = 45)

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq) | **0.978** |
| Fine-tuned DistilBERT | **0.756** |
| Difference | −0.222 (fine-tuning regression) |

The zero-shot LLM baseline **outperformed** fine-tuned DistilBERT by 22 percentage points. On this task, a general-purpose LLM with clear label definitions handled sentiment classification better than a small model trained on 300 examples.

### Per-class metrics

#### Zero-shot baseline (Groq) — test set n = 45

| Label | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| negative | 1.00 | 0.96 | 0.98 | 23 |
| positive | 0.96 | 1.00 | 0.98 | 22 |
| **Accuracy** | | | **0.98** | **45** |
| macro avg | 0.98 | 0.98 | 0.98 | 45 |
| weighted avg | 0.98 | 0.98 | 0.98 | 45 |

The baseline missed only **1 of 45** reviews (one negative review classified as positive). Both classes reached F1 ≈ 0.98.

#### Fine-tuned DistilBERT — test set n = 45

| Label | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| negative | 0.83 | 0.65 | 0.73 | 23 |
| positive | 0.70 | 0.86 | 0.78 | 22 |
| **Accuracy** | | | **0.76** | **45** |
| macro avg | 0.77 | 0.76 | 0.75 | 45 |
| weighted avg | 0.77 | 0.76 | 0.75 | 45 |

The fine-tuned model made **11 errors** (34/45 correct). It over-predicted `positive` for negative reviews (8 false positives) while missing fewer positive reviews (3 false negatives). Confidence on wrong predictions was low (~0.50–0.56), suggesting the model was uncertain but still guessed wrong.

### Confusion matrix — zero-shot baseline (Groq)

Derived from per-class recall (45/45 parseable responses):

|  | Predicted negative | Predicted positive |
|--|------------------:|-------------------:|
| **True negative** | 22 | 1 |
| **True positive** | 0 | 22 |

### Confusion matrix — fine-tuned DistilBERT

|  | Predicted negative | Predicted positive |
|--|------------------:|-------------------:|
| **True negative** | 15 | 8 |
| **True positive** | 3 | 19 |

See also `evaluation/confusion_matrix.png` in this repo (exported from Colab).

---

### Wrong predictions — analysis

**11 wrong out of 45.** Below are three analyzed in depth:

**1. Sarcastic wordplay (true: negative → predicted: positive, confidence: 0.51)**  
> *"Ming The Merciless does a little Bardwork and a movie most foul!"*

The review uses Shakespearean punning ("Bardwork") to mock the film. DistilBERT likely latched onto literary/playful phrasing without grasping that "most foul" signals disapproval. The Groq baseline understood the ironic tone; the fine-tuned model treated unusual vocabulary as neutral-to-positive. **Pattern:** short, witty negative reviews that don't use obvious insult words.

**2. Opening sounds negative, verdict is positive (true: positive → predicted: negative, confidence: 0.54)**  
> *"At first glance I expected this film to be crappy because I thought the plot would be so excessively feminist. But I was wrong. As you maybe have read in earlier published comments, I agree in that th..."*

The review opens with low expectations and criticism, then pivots to agreement with positive comments. DistilBERT weighted the opening negative framing more heavily than the reversal ("But I was wrong"). **Pattern:** reviews with delayed or reversed sentiment — structure signals one label, overall verdict signals another.

**3. Starts positive, turns negative (true: negative → predicted: positive, confidence: 0.52)**  
> *"This movie started out good, i felt like i was watching an adult version of Seinfeld. Much to quickly i started questioning the situations and actions of the main characters, and found no answers to w..."*

The reviewer begins with praise ("started out good") before the review turns critical. The model appears to anchor on early positive phrases rather than the overall negative conclusion. **Pattern:** mixed-tone reviews where negative sentiment dominates but positive words appear early — same boundary issue as in planning.md.

**Systematic pattern:** 8 of 11 errors were negative reviews predicted as positive. The model has a **positive bias** — it defaults toward `positive` when cues are ambiguous, especially with low confidence (~0.50–0.56 on all misclassifications).

---

### Sample classifications (fine-tuned model)

| Review (truncated) | Predicted | Confidence | Notes |
|--------------------|-----------|------------|-------|
| "An absolute masterpiece. Every scene is beautifully shot..." | positive | ~0.95 | **Correct** — clear praise vocabulary; high confidence is reasonable. |
| "Terrible movie. Boring plot, bad dialogue..." | negative | ~0.92 | Correct — unambiguous negative language. |
| "The visuals were great but the story fell apart..." | negative | ~0.61 | Often wrong — mixed tone; model latches onto criticism. |
| "Funny, heartfelt, and well-paced. One of my favorites." | positive | ~0.88 | Correct — straightforward positive review. |
| "I don't know what people see in this film. Overrated." | negative | ~0.71 | Correct — dismissive tone despite no extreme words. |

*Confidence values from Colab Section 7 sample-classification cell.*

---

## Reflection: What the Model Learned vs. What We Intended

**Intended:** The model should learn overall review sentiment — whether the reviewer recommends the film or warns others away — including handling sarcasm and mixed reviews via the overall verdict rule.

**Learned:** DistilBERT largely learned **surface sentiment cues** (positive/negative word lists) rather than pragmatic overall judgment. It works well on clear, unambiguous reviews but fails when tone is ironic, mixed, or understated. The Groq baseline succeeded because LLMs pre-trained on vast text have stronger pragmatic and contextual understanding of sentiment, even zero-shot.

**Gap:** Fine-tuning on 300 examples did not overcome the base model's limited world knowledge. The −22 point regression vs. baseline shows that for straightforward sentiment on well-formed reviews, task-specific fine-tuning added little — and may have **overfit** to lexical patterns in the small training slice.

---

## Spec Reflection

**How the spec helped:** Requiring a baseline comparison revealed that fine-tuning was unnecessary for this task — without the Groq comparison, 76% accuracy might have seemed acceptable in isolation.

**How implementation diverged:** The spec suggested choosing a niche community (subreddit, Discord) with custom labels. I used IMDb sentiment instead for a ready-made dataset. This satisfied the technical requirements but made the task easier for the LLM baseline and harder to justify fine-tuning gains — a tradeoff I accepted for speed.

---

## AI Usage

1. **Label design & planning:** Used Cursor/Claude to stress-test binary sentiment labels with borderline review examples. Revised the "overall recommendation" tie-break rule after several generated examples could not be labeled cleanly without it.

2. **Notebook configuration:** AI assistance configured the Colab notebook for IMDb (`stanfordnlp/imdb`), wrote the Groq system prompt, and fixed the `HfUriError` by switching from `imdb` to the namespaced dataset ID.

3. **Annotation:** Did **not** manually label 300 reviews. Labels come from the HuggingFace IMDb dataset; I sampled a balanced subset and verified distribution.

4. **Failure analysis:** Used AI to identify error patterns (sarcasm, mixed tone, short reviews) in misclassified examples; verified patterns against notebook Section 4 output.

---

## Repository Contents

| File | Purpose |
|------|---------|
| `planning.md` | Design doc (labels, edge cases, metrics, AI plan) |
| `data/imdb_labeled.csv` | 300 labeled reviews |
| `Copy_of_ai201_project3_takemeter_starter_clean.ipynb` | Colab training + evaluation notebook |
| `evaluation/evaluation_results.json` | Exported metrics |
| `evaluation/confusion_matrix.png` | Confusion matrix image (from Colab) |
| [Demo video](https://streamable.com/6juaxa) | 3–5 min walkthrough of classifications and evaluation report |

---

## How to Run

### Generate dataset (local, optional)

```powershell
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\python scripts\prepare_imdb_dataset.py
```

### Train & evaluate (Colab)

1. Upload the notebook to Google Colab.
2. **Runtime → Change runtime type → T4 GPU**
3. Add `GROQ_API_KEY` in Colab Secrets.
4. **Runtime → Run all**
5. Download `evaluation_results.json` and `confusion_matrix.png` into the `evaluation/` folder.

---

## Demo Video

**[Watch TakeMeter demo →](https://streamable.com/6juaxa)**

The video shows:
- 5 reviews classified by the fine-tuned model with label and confidence
- One correct prediction explained (clear negative language)
- One incorrect prediction explained (sarcastic wordplay misclassified as positive)
- A brief walkthrough of the evaluation report (97.8% baseline vs 75.6% fine-tuned)
