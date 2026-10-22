# Conspiracy Leaders: A Dataset of Behavioral and Psycholinguistic Precursors to Conspiracy Community Founding
---

## Overview

Online conspiracy theories spread rapidly through community-driven ecosystems, yet existing computational research primarily examines established groups rather than the individuals who initiate them. This project investigates the behavioral, relational, and psycholinguistic precursors of conspiracy community founders on Reddit during the months preceding community creation.

To isolate conspiratorial dynamics from general community-building behaviors, this repository implements an end-to-end data collection, matching, and feature extraction pipeline. It pairs creators of conspiracy subreddits (Anchors) with statistical twins who founded non-conspiratorial communities (Controls) under comparable platform conditions. The resulting dataset comprises 196 verified founders (98 balanced 1:1 pairs) characterized across 124 behavioral, network, psycholinguistic, and semantic attributes.

---

## Pipeline Architecture

The methodological workflow is organized into four sequential phases corresponding to the core modules of this repository:

### 1. Founder Identification and Matching (`01_matching/`)
* **Subreddit and Moderator Harvesting:** Queries the ArcticShift API (via `aiohttp` with rate-limiting and exponential backoff) and Reddit's `/about/moderators` endpoint to retrieve community creation timestamps ($t_{sub}$) and moderator appointment timestamps ($t_{mod}$).
* **Founder Filtering:** Identifies a user $u$ as a community founder if appointed moderator within one hour of subreddit creation ($t_{mod}(u) \le t_{sub} + 3600$), filtering out deleted/inactive accounts and enforcing at least 6 months of platform activity.
* **Cohort Definition:** Projects communities onto a latent conspiracy axis based on the Waller-Anderson (WA) community embeddings. Founders in the top 25% of the WA scale form the Anchor cohort ($A$), while founders in the bottom 50% form the Candidate control pool ($C$), excluding the intermediate 25% to reduce classification noise.
* **Three-Stage Quasi-Experimental Matching:** Extracts baseline features strictly from the pre-observation window ($t < t_{founding} - 3\text{ months}$) and filters pairs sequentially:
  1. *Exact Matching* on the calendar year of the user's first recorded activity (`first_activity_date`).
  2. *Coarsened Exact Matching (CEM)* on pre-observation contribution volume (`total_contributions`) across six activity tiers.
  3. *Word Mover's Distance (WMD)* computed via `Gensim` over synthetic documents built from each user's historical `subreddit_contribution_histogram`.
* **Global Optimization:** Formulates the bipartite assignment as a Linear Programming (LP) problem in `PuLP` and solves it via the COIN-OR Branch and Cut (`CBC`) solver to maximize valid matches while minimizing aggregate WMD, yielding 98 disjoint 1:1 pairs.

### 2. Behavioral Recording and Network Reconstruction (`02_behavior_recording/`)
* **Observation Window Corpus Extraction:** Harvests all founder activity within the 90-day window preceding subreddit creation ($W_{obs} = [t_{founding} - 90\text{ days}, t_{founding}]$) from the ArcticShift archive across four categories: Authored Posts, Authored Comments, Commented Posts, and locally reconstructed Received Comments.
* **Acquaintance Interaction Graph:** Reconstructs bilateral conversational ties (inbound and outbound replies) between each founder and their acquaintances during $W_{obs}$, excluding self-replies and deleted accounts.
* **Current Conspiracist (CC) Classification:** Collects pre-observation subreddit histograms ($t < t_{pre}$) for all acquaintances and computes a contribution-weighted conspiracy score $\sigma(v)$ on the Waller-Anderson scale[cite: 1]. Acquaintances with $\sigma(v) \ge 0.15$ are classified as Current Conspiracists (CC)[cite: 1].

### 3. Multi-Dimensional Feature Engineering (`03_feature_engineering/`)
* **Ratio Features (18 attributes):** Captures activity allocation (`Posting Ratio`, `Commenting Ratio`), moderation friction and self-censorship (`Moderated Ratio`, `Deleted Ratio`), audience endorsement (`Positive Score Ratio`, `Negative Score Ratio`, net accumulated karma), bilateral peer exposure (`Outbound CC Ratio`, `Inbound CC Ratio`, `Combined CC Share`, `Global Reception Ratio`), thread co-presence (`Thread Exposure`), and community specialization via the Gini index ($G_{obs}$) and Growing Exclusivity ($\Delta G = G_{obs} - G_{pre}$)[cite: 1].
* **Psychological, Affective, and Syntactic Markers (80 attributes):** Evaluates authored text across three subsets (Global activity, CC-directed replies, and Non-CC-directed replies)[cite: 1]:
  * *Empath (72 attributes):* 24 standalone lexical categories covering negative affect, hostility/conflict, positive affect/prosociality, and interpersonal power[cite: 1].
  * *VADER (3 attributes):* Mean compound sentiment polarity across global, CC, and Non-CC interactions[cite: 1].
  * *spaCy (5 attributes):* Named entity mentions (`PERSON` and `ORG` via `en_core_web_sm`) and pronoun-to-token ratios (`PRON`)[cite: 1].
* **Linguistic Coordination (9 attributes):** Measures stylistic accommodation as the cosine similarity between a founder's reply vector and the acquaintance's preceding message vector across Global ($\mathbb{R}^{25}$), Positive Affect ($\mathbb{R}^{7}$), and Negative Affect ($\mathbb{R}^{11}$) feature spaces[cite: 1].
* **Content Conspiracy Similarity (1 attribute):** Encodes founder contributions and a benchmark corpus of the top 100 WA-ranked conspiracy subreddits using Sentence-BERT (`all-mpnet-base-v2`, 768 dimensions), computing the mean cosine similarity between the founder's semantic centroid and benchmark community centroids[cite: 1].

### 4. Dataset Assembly and Empirical Analysis (`04_analysis/`)
* **Relational Schema Integration:** Compiles the final tabular dataset ($N = 196$ founders, 124 columns) preserving the 1:1 pairing structure through explicit relational keys (`is_anchor` and `user_matched`) alongside historical metadata[cite: 1].
* **Moderation Analysis:** Implements a logistic regression interaction model evaluating how linguistic coordination with conspiracy peers (`avg_linguistic_coordination_cc`) moderates the relationship between pre-founding CC interaction volume (`ratio_interactions_cc`) and the likelihood of founding a conspiracy community ($\beta_3 = 14.56, p = 0.026$)[cite: 1].

---

## Repository Structure

```text
.
├── config.yaml                 # Project paths, thresholds, and feature settings
├── config_loader.py            # Loads the project configuration
├── Makefile                    # Common data-processing commands
├── pyproject.toml              # Python project metadata and tooling configuration
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                    # Original founder and Reddit activity data
│   ├── external/               # External embeddings and subreddit reference data
│   ├── interim/                # Intermediate matching, scoring, and feature outputs
│   └── processed/              # Final processed dataset
├── src/
│   ├── data/                   # Data collection, extraction, and dataset assembly
│   ├── feature/                # Matching, scoring, and feature engineering
│   └── old/                    # Earlier data-cleaning and extraction scripts
├── notebooks/                  # Exploratory analyses, visualizations, and result checks
├── interaction plot.png        # Exported interaction-effect visualization
└── README.md
```

The main processing code is organized under `src/data/` and `src/feature/`. Data moves
from the raw collection in `data/raw/` through intermediate artifacts in
`data/interim/` to the final table in `data/processed/`. For privacy reason, the data folder has not been committed. 
The notebooks contain exploratory analyses and visualizations built from these outputs.
