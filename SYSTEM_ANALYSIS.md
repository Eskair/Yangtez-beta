# Yangtze AI Reviewer — System Analysis

> A structured, presentation-ready analysis of the AI-based research proposal review system.

---

## 1. System Architecture

The system has **two parallel pipelines** and **one optional retrieval module**, all sharing a common document preparation layer.

### Pipeline A — Primary Review (`run_review.py` / `app.py`)

This is the user-facing Gradio application. It produces a single Markdown review report.

```
┌──────────────────────────────────────────────────────────────────┐
│                      PIPELINE A (Primary)                        │
│                                                                  │
│  PDF/DOCX ──► Stage 0: Prepare ──► Domain Profile ──► Metrics   │
│                  │                     │                  │      │
│                  ▼                     ▼                  ▼      │
│              pages.json          profile.json      metric_report │
│                  │                     │                  │      │
│                  └────────┬────────────┘──────────────────┘      │
│                           ▼                                      │
│                  9 Review Tasks (keyword + heuristic scoring)    │
│                           │                                      │
│                           ▼                                      │
│                  Dimension Aggregation ──► Verdict Computation   │
│                           │                        │             │
│                           ▼                        ▼             │
│                  Quality Notes (LLM)     PRIORITY_SUPPORT /      │
│                           │              SUPPORT / CONDITIONAL / │
│                           ▼              CONCERN                 │
│                  Final Report (LLM) ──► Markdown Report          │
└──────────────────────────────────────────────────────────────────┘
```

### Pipeline B — Extended Offline Research Pipeline (`run_pipeline.py`)

A deeper, research-oriented workflow with multiple LLM passes.

```
┌──────────────────────────────────────────────────────────────────┐
│                   PIPELINE B (Extended / Offline)                 │
│                                                                  │
│  Stage 0: Prepare                                                │
│       ▼                                                          │
│  Stage 1: Extract Facts (LLM) ──► raw_facts.jsonl               │
│       ▼                                                          │
│  Stage 2: Build Dimensions (LLM) ──► dimensions_v2.json         │
│       ▼                                                          │
│  Stage 2.5: Domain Profile (LLM)                                 │
│       ▼                                                          │
│  Stage 3: Generate Questions (deterministic)                     │
│       ▼                                                          │
│  Stage 4: LLM Answering (OpenAI + DeepSeek, multi-model)        │
│       ▼                                                          │
│  Stage 5: Post-Processing (heuristic scoring, no LLM)            │
│       ▼                                                          │
│  Stage 6: AI Expert Opinion (LLM + local fallback)               │
│       ▼                                                          │
│  Stage 7: Final Report Assembly ──► Markdown Report              │
└──────────────────────────────────────────────────────────────────┘
```

### Module C — Optional Web Retrieval (standalone scripts)

```
  Parsed Dimensions + Questions
       ▼
  LLM-Generated Search Queries ──► Web Search (Google / Tavily / DDG)
       ▼
  Embedding Clustering + Fusion (BGE-M3) ──► fused_evidence/
       ▼
  Chroma Vector DB (BGE-large-zh)
```

This module is **not integrated** into either main pipeline by default.

---

## 2. Data Flow

### Pipeline A — Primary (Gradio UI / CLI)

```
Input File (.pdf/.docx/.txt)
    │
    ▼
┌─── Stage 0: prepare_text() ──────────────────────────────────┐
│  • pdfplumber text extraction                                 │
│  • Optional: OCR via Tesseract                                │
│  • Optional: Vision LLM per-page layout reconstruction        │
│  • Optional: Table Transformer for table detection            │
│  OUTPUT: pages.json, full_text.txt, [page_semantics.json]     │
└───────────────────────────────────────────────────────────────┘
    │
    ├──► Domain Profiler (heuristic OR LLM JSON)
    │      OUTPUT: profile with domain, methods, risks, terminology,
    │              evaluation_focus, document_form
    │
    ├──► Metric Checker (pure regex/heuristic)
    │      OUTPUT: evidence_level (strong/moderate/weak),
    │              numeric spans, missing signals, conflicts
    │
    ▼
┌─── 9 Review Tasks (loop) ────────────────────────────────────┐
│  For each task (problem, objectives, methods, evidence,       │
│  feasibility, innovation, risks, team, outcomes):             │
│                                                               │
│  1. build_specialized_question() — inject profile slots       │
│  2. select_evidence() — keyword-match pages, apply multiplier │
│  3. score_task() — heuristic base=5.0 ± feature bonuses      │
│  4. build_task_assessment() — template judgment + evidence     │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
  Aggregate: dimension_scores → compute_final_verdict()
  (arithmetic mean of score_10 values → threshold → verdict)
    │
    ├──► LLM Quality Notes (optional self-check)
    │
    ▼
  LLM Final Report Generation
  (massive prompt with verdict lock, page citation allowlists,
   evidence digest, quality notes, staleness warnings)
    │
    ▼
  OUTPUT: review.json + {proposal}_review_report.md
```

### Pipeline B — Extended

```
full_text.txt
    │
    ├──► LLM Fact Extraction (chunked) ──► raw_facts.jsonl
    │
    ├──► LLM Dimension Grouping ──► dimensions_v2.json (5 dimensions)
    │
    ├──► Deterministic Question Generation (templates + profile)
    │       ──► generated_questions.json
    │
    ├──► Multi-Model LLM Answering (OpenAI + DeepSeek variants)
    │       ──► all_refined_items.json (candidates per question)
    │
    ├──► Post-Processing (heuristic scoring: length, claims,
    │    structure, alignment, consistency, penalties)
    │       ──► final_payload.json, metrics.json
    │
    ├──► AI Expert Opinion (LLM or local rules)
    │       ──► ai_expert_opinion.json/md, GO/HOLD/NO-GO
    │
    ▼
  Final Report Assembly ──► {pid}_final_report.md
```

---

## 3. Key Strengths

### 3.1 Thoughtful Multi-Layer Design
The system decomposes proposal review into nine well-chosen review tasks mapped to five evaluation dimensions (team, objectives, strategy, innovation, feasibility). This mirrors how human reviewers actually evaluate proposals and produces structured, interpretable output.

### 3.2 Domain-Adaptive Profiling
The domain profiler extracts a structured profile (domain, methods, risks, terminology, document form) and injects it into review prompts via slot-based templates. This adapts generic review logic to specific proposals — a significant advantage over one-size-fits-all approaches.

### 3.3 Evidence Grounding and Citation Control
The system implements a page-level evidence selection mechanism with allowlists. The final LLM prompt explicitly forbids invented page numbers and enforces row-by-row correspondence between issues and suggestions. This is a sophisticated anti-hallucination guard.

### 3.4 Multi-Model Consistency Checking (Pipeline B)
The extended pipeline queries multiple LLM providers (OpenAI + DeepSeek), scores candidates on multiple axes (length, claims, structure, alignment, consistency), and uses pairwise Jaccard similarity + contradiction detection. This "best-of-N" approach with cross-model validation is a strong reliability mechanism.

### 3.5 Graceful Degradation
Every LLM-dependent component has a fallback: the profiler falls back to heuristics, the expert opinion falls back to local rules, the scoring is entirely non-LLM. The system can produce a review even without API access.

### 3.6 Metric-Aware Prompting
The metric checker detects quantitative signals (money, timelines, percentages, validation terms) and feeds guidance into downstream prompts. This ensures the LLM's review is calibrated to the actual evidence strength.

### 3.7 Professional UI
The Gradio frontend provides a clean upload-and-review experience with real-time pipeline status, error handling with actionable hints, and copy/download actions.

---

## 4. Major Weaknesses

### 4.1 Scoring Is Fundamentally Disconnected from Content Understanding

**The core problem.** In Pipeline A, task scores are computed from keyword matching and binary feature flags — not from any understanding of the proposal's actual quality.

```python
# Actual scoring logic (run_review.py)
base = 5.0
evidence_strength = min(2.0, sum(min(ev.score, 3.0) for ev in evidences) / 6.0)
score = base + evidence_strength  # → range ~5.0 to 7.0
```

A score of "7.2 / 10 for innovation" does not mean the proposal is innovative. It means the system found keyword matches on pages that mention "领先" or "首台". Two radically different proposals on the same topic could receive nearly identical scores.

### 4.2 Verdict Thresholds Are Arbitrary and Fragile

The verdict is determined by hard-coded thresholds on the arithmetic mean of heuristic scores:

| Condition | Verdict |
|-----------|---------|
| mean ≥ 8.55 and conf ≥ 0.76 | PRIORITY_SUPPORT |
| mean ≥ 7.68 and conf ≥ 0.69 | SUPPORT |
| mean ≥ 6.1 | CONDITIONAL |
| mean < 5.85 | CONCERN |

Given that the base score is 5.0 and modifiers are small (±1.2 max), nearly all proposals will cluster in the 5.0–7.5 range, landing in CONDITIONAL. The thresholds for PRIORITY_SUPPORT (≥ 8.55) are essentially unreachable with the current scoring formula.

### 4.3 The LLM Report and the Machine Scores Describe Different Realities

The system computes a verdict using heuristic scores, then **forces** the LLM to endorse it:

> "评审结论一栏必须逐字填写为：**{vlabel}**（系统判定），禁止使用英文代号，不得擅自改成其他档位用语。"

The LLM has no ability to override a verdict it may disagree with based on the actual content. This creates a split-brain architecture: the narrative portion reflects LLM understanding, but the conclusion is dictated by keyword statistics.

### 4.4 Hallucination Risk in Final Report Generation

Despite the page citation allowlists, the LLM receives a massive prompt (~3000+ tokens of instructions alone) and must produce a formal eight-section review. The evidence digest provides only 2 snippets per task (≤80 chars each). The LLM is expected to write detailed strengths, issues, and suggestions with this thin grounding — a setup that invites confabulation.

### 4.5 Evidence Selection Is Shallow

`select_evidence()` scores pages by counting keyword hits with fixed weights:
- Task keyword match: +2.0
- Profile term match: +1.2
- Numeric density: +0.15/number (capped at 2.0)
- Title/role regex for team: +5.5

There is no semantic similarity, no embedding-based retrieval, and no cross-page reasoning. A page about "风险管理框架" (risk management framework) would score low for the "risks" task if it doesn't contain the exact keyword "风险".

### 4.6 Two Disconnected Pipelines with No Shared Intelligence

Pipeline A and Pipeline B are architecturally independent. Pipeline A's keyword-based scoring and Pipeline B's multi-model Q&A system do not share findings, scores, or intermediate results. A user running the Gradio app gets none of Pipeline B's deeper analysis.

### 4.7 Dead Code

`src/main.py` imports modules that don't exist (`backend.chains.base_chain`, `backend.chains.orchestrator`). This is orphaned code from a previous architecture.

### 4.8 Template Judgments Regardless of Content

`build_task_assessment()` uses static template strings for strengths and weaknesses:

```python
TASK_TEMPLATES = {
    "problem": (
        "The proposal appears to target a concrete problem area...",
        "The motivation would be stronger with clearer articulation...",
    ),
    ...
}
```

Every proposal receives the same boilerplate observations, regardless of actual content quality.

---

## 5. High-Impact Improvements

### 5.1 Replace Keyword Scoring with LLM-Based Evaluation

**Impact: Transformative.** The single highest-value change.

Instead of computing scores from keyword counts, have the LLM evaluate each dimension directly:

```
For each of the 9 review tasks:
  1. Select evidence pages (keep current method or add embeddings)
  2. Send task prompt + evidence text to LLM
  3. LLM returns structured JSON: {score: 1-10, confidence: 0-1,
     judgment: "...", strengths: [...], weaknesses: [...]}
  4. Aggregate LLM scores across tasks → verdict
```

This makes scores reflect actual content understanding. The verdict and the narrative become consistent because the same intelligence produces both.

### 5.2 Add Embedding-Based Evidence Retrieval

**Impact: High.** The infrastructure already exists in the codebase (BGE embeddings for Pipeline C) but is not used in Pipeline A.

Replace or augment keyword evidence selection with:
1. Embed all pages at preparation time
2. Embed each review task description
3. Retrieve top-k pages by cosine similarity
4. Combine with keyword scores for hybrid retrieval

### 5.3 Unify the Two Pipelines

**Impact: High.** Merge Pipeline B's multi-model Q&A and post-processing into Pipeline A as optional stages.

The Gradio app should offer a "Quick Review" (current Pipeline A, ~2 min) and "Deep Review" (Pipeline A + B's LLM Q&A + expert opinion, ~10 min) mode.

### 5.4 Ground the LLM Report in Richer Evidence

**Impact: Medium-High.** Currently the LLM report receives 2 snippets × 80 chars per task.

Instead, pass the full evidence page text (up to a token budget) so the LLM can write substantive observations. Use a chunked approach if the context window is limited.

### 5.5 Add Human-in-the-Loop Calibration

**Impact: Medium.** Collect human reviewer scores on a calibration set. Use these to:
- Validate that LLM-based scoring (improvement 5.1) correlates with expert judgment
- Tune verdict thresholds empirically rather than by intuition
- Identify systematic biases (e.g., does the system overrate team sections?)

---

## 6. Final Upgraded Design

A clean, production-level architecture that addresses the core weaknesses while preserving the system's existing strengths.

```
┌──────────────────────────────────────────────────────────────────┐
│                  UPGRADED SYSTEM ARCHITECTURE                     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LAYER 1: DOCUMENT UNDERSTANDING                        │     │
│  │                                                         │     │
│  │  PDF → Text + OCR + Layout → pages.json + full_text     │     │
│  │  Embed all pages (BGE / OpenAI embeddings)              │     │
│  │  Domain Profile (LLM, cached)                           │     │
│  │  Metric Signals (regex, unchanged)                      │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LAYER 2: EVIDENCE RETRIEVAL (hybrid)                   │     │
│  │                                                         │     │
│  │  For each of 9 review tasks:                            │     │
│  │    • Keyword scoring (existing, fast)                   │     │
│  │    • Embedding similarity (new, accurate)               │     │
│  │    • Reciprocal rank fusion → top-k pages               │     │
│  │    • Extract full text of selected pages                │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LAYER 3: LLM-BASED EVALUATION (new core)              │     │
│  │                                                         │     │
│  │  For each task, send to LLM:                            │     │
│  │    • Task prompt (domain-adapted, as today)             │     │
│  │    • Full evidence page text (not 80-char snippets)     │     │
│  │    • Metric guidance suffix (as today)                  │     │
│  │                                                         │     │
│  │  LLM returns structured JSON:                           │     │
│  │    { score_10, confidence, judgment,                    │     │
│  │      strengths[], weaknesses[],                         │     │
│  │      cited_pages[], missing_info[] }                    │     │
│  │                                                         │     │
│  │  Optional: run 2 models, cross-validate scores          │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LAYER 4: AGGREGATION & VERDICT                         │     │
│  │                                                         │     │
│  │  • Aggregate LLM scores by dimension (weighted mean)    │     │
│  │  • Compute confidence from LLM self-reported + cross-   │     │
│  │    model agreement (if 2+ models used)                  │     │
│  │  • Apply calibrated thresholds (from human data)        │     │
│  │  • Quality self-check pass (existing, keep)             │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LAYER 5: REPORT GENERATION                             │     │
│  │                                                         │     │
│  │  LLM synthesizes final report from:                     │     │
│  │    • Per-task structured evaluations (from Layer 3)     │     │
│  │    • Dimension scores and verdict (from Layer 4)        │     │
│  │    • Evidence citations (grounded, not invented)        │     │
│  │                                                         │     │
│  │  Key: verdict and narrative are now CONSISTENT          │     │
│  │  because the same LLM produced both the scores          │     │
│  │  and the reasoning behind them.                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│  OUTPUT: review.json + Markdown report                           │
│  (Gradio UI with Quick/Deep mode toggle)                         │
└──────────────────────────────────────────────────────────────────┘
```

### What Changes vs. Current System

| Component | Current | Upgraded |
|-----------|---------|----------|
| Evidence retrieval | Keyword only | Keyword + embedding hybrid |
| Task scoring | Heuristic (base 5.0 + keyword bonuses) | LLM-evaluated (structured JSON) |
| Verdict source | Arithmetic mean of heuristic scores | Aggregated LLM scores, calibrated thresholds |
| Report grounding | 80-char snippets, forced verdict | Full evidence text, consistent verdict |
| Pipeline integration | Two separate pipelines | Unified with Quick/Deep modes |
| Score–narrative consistency | Disconnected (heuristic score, LLM text) | Unified (LLM produces both) |

### What Stays the Same

- Stage 0 document preparation (robust, well-designed)
- Domain-adaptive profiling and slot injection
- Metric checker for quantitative signal detection
- Page citation allowlists for anti-hallucination
- Quality self-check pass
- Gradio UI structure
- Multi-model cross-validation concept (from Pipeline B)
- Local fallback when LLM is unavailable

### Cost and Latency Implications

The upgraded design adds ~9 LLM calls (one per review task for scoring) on top of the existing 1–2 calls (profile + final report). At `gpt-4o-mini` pricing, this adds approximately $0.02–0.05 per review. Latency increases by ~30–60 seconds if calls are sequential, or ~10 seconds if parallelized. This is acceptable for a review system where quality matters far more than speed.

---

## Summary

The Yangtze AI Reviewer demonstrates sophisticated architectural thinking — domain-adaptive prompting, multi-model cross-validation, metric-aware guidance, and structured anti-hallucination controls. Its primary weakness is the disconnect between heuristic scoring and LLM-generated narratives: the scores don't reflect content understanding, and the verdict is forced onto the LLM. The single most impactful improvement is replacing keyword-based scoring with LLM-based structured evaluation, which would unify the scoring and narrative layers into a coherent system suitable for production use.
