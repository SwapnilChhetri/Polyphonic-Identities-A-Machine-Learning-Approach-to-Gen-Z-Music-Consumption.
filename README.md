# Polyphonic Identities: A Machine Learning Approach to Gen Z Music Consumption

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-green)
![Spotify API](https://img.shields.io/badge/Spotify%20Web%20API-Data%20Mining-1DB954?logo=spotify&logoColor=white)
![Status](https://img.shields.io/badge/MSc%20Dissertation-Completed%20(May%202025)-success)

> **Executive Portfolio Summary:** This repository presents the research, architectural design, and analytical findings of my MSc in Data Analytics dissertation at **London Metropolitan University (Islington College)**. The project develops a multi-modal data analytics and machine learning framework—processing **$>1.9\text{ million}$ streaming records** across **73 countries**—to disentangle genuine multicultural music exploration among Generation Z from algorithm-driven homogenization.

---

## Table of Contents
1. [Executive Summary & Business Context](#1-executive-summary--business-context)
2. [Core Objectives & Analytical Questions](#2-core-objectives--analytical-questions)
3. [Data Architecture & Engineering Pipeline](#3-data-architecture--engineering-pipeline)
4. [Methodology & Mathematical Frameworks](#4-methodology--mathematical-frameworks)
5. [Key Analytical Findings & Insights](#5-key-analytical-findings--insights)
6. [Machine Learning: Multi-Class Genre Classification](#6-machine-learning-multi-class-genre-classification)
7. [Preserved Code Architecture (Python OOP Ingestion)](#7-preserved-code-architecture-python-oop-ingestion)
8. [Product & Industry Recommendations](#8-product--industry-recommendations)
9. [Repository Structure & Code Availability Note](#9-repository-structure--code-availability-note)
10. [Author & Contact](#10-author--contact)

---

## 1. Executive Summary & Business Context

In the modern streaming economy, digital platforms like Spotify ($626\text{ million}$ monthly active users) and TikTok provide borderless access to global music, enabling genres like Afrobeats and K-Pop to dominate Western charts. However, proprietary recommendation engines often optimize for immediate engagement rather than serendipitous discovery, risking the creation of algorithmic "echo chambers."

This project investigates how Generation Z constructs **"Polyphonic Identities"** (complex, blended musical tastes spanning multiple cultures and genres). By combining macro-level global chart datasets ($1955\text{--}2024$), cross-platform virality metrics (Spotify, YouTube, TikTok), and micro-level extended user streaming logs, this study delivers:
* **Longitudinal Trend Analysis:** Quantifying 70 years of sonic evolution across 11 acoustic dimensions.
* **Cross-Platform Attribution:** Determining whether social media virality (TikTok/YouTube) or internal platform curation (Spotify playlist reach) drives global chart rankings.
* **Graph-Theoretic Taxonomy:** Mapping a weighted Genre Co-occurrence Network using community detection to identify mainstream hubs and cross-genre "bridge" styles.
* **Information-Theoretic User Profiling:** Formulating a normalized Shannon Entropy metric (**Polyphonic Score**) alongside behavioral telemetry (skip rates $<30\text{ s}$, query logs, release recency) to segment listener archetypes.
* **Predictive Modeling:** Engineering an **XGBoost Classifier** that outperforms baseline audio-only models by **$+41.41\%$ accuracy** through the integration of contextual and behavioral features.

---

## 2. Core Objectives & Analytical Questions

| Analytical Pillar | Industry / Product Question | Analytical Approach |
| :--- | :--- | :--- |
| **Sonic Homogenization** | Have global hit songs become sonically uniform despite diverse cultural origins? | Longitudinal feature tracking ($1955\text{--}2024$) & multi-dimensional radar profiling across genres. |
| **Cross-Platform Attribution** | Does TikTok virality translate into sustained Spotify chart dominance? | Pearson correlation matrices & Granger-causality evaluation across Spotify, TikTok, and YouTube metrics. |
| **Algorithmic Mediation vs. Agency** | How do active searchers differ from passive, algorithm-reliant listeners in retention and skip behavior? | Granular telemetry analysis of `reason_start` logs, $<30\text{ s}$ skip rates, and `SearchQueries.json` archives. |
| **Quantifying Taste Diversity** | How can platforms mathematically measure the breadth and balance of a user's genre consumption? | Normalized Shannon Entropy (**Polyphonic Score**) & undirected NetworkX graph modularity. |
| **Genre Prediction at Scale** | Can machine learning classify fluid modern genres using acoustic features alone, or is behavioral context required? | Comparative modeling: Baseline Random Forest vs. Feature-Engineered XGBoost across 32 target classes. |

---

## 3. Data Architecture & Engineering Pipeline

The study engineered a hybrid **Python + R** data pipeline to ingest, harmonize, and merge heterogeneous datasets spanning macro-level industry charts and micro-level user telemetry.

### 3.1 Quantitative & Qualitative Data Sources
1. **Spotify Web API Extractions (`spotipy`):** Programmatic extraction of track metadata, artist URIs, popularity scores ($1\text{--}100$), and 11 standardized audio features across $40+$ Gen Z-focused playlists.
2. **Daily Country Charts (`TopByCountries`):** $>1.9\text{ million}$ records covering daily top charts across 73 countries, enabling geographic and cultural segmentation.
3. **Cross-Platform Virality Compilation (`MostStreamed 2024`):** $\sim 4,260$ top tracks enriched with multi-platform telemetry (Spotify streams/playlist reach, YouTube views/likes, TikTok posts/views/likes, Shazam counts).
4. **Historical Baseline (`Maindata_Merged`):** Unified master dataset combining random historical samples ($1950\text{s--Present}$), the Top $10,000$ Billboard/ARIA tracks, and contemporary youth playlists.
5. **Granular User Telemetry & Survey Data:** Complete GDPR/privacy-compliant Spotify Account & Extended Streaming archives from 5 anonymized Gen Z cohorts (`SWC0`, `SWC1`, `AND0`, `ANR0`, `ANS0`) paired with $N = 27$ structured survey responses.

### 3.2 Preprocessing & Harmonization Workflow (R & Python)
* **Schema Standardization & Deduplication:** Standardized all attributes to `snake_case`. Global deduplication on `track_id` retained only the record with the maximum Spotify popularity score to preserve representative metadata.
* **Temporal Rolling Joins (`data.table`):** Aligned weekly global chart snapshots (`GlobalWeekly`) with daily country records (`TopByCountries`) using nearest-neighbor rolling joins to eliminate temporal coverage gaps.
* **Outlier Mitigation (Winsorization):** Generated dual dataset variants—retaining raw extreme values (e.g., loudness $> 0\text{ dB}$) for descriptive fidelity while capping continuous variables at $1.5 \times \text{IQR}$ (`maindata_winsorize`) to stabilize machine learning and clustering workflows.
* **HashMap Indexing:** Implemented in-memory dictionary/HashMap indexing for user JSON bundles and track URI lookups, reducing query execution time by **$\sim 80\%$**.

---

## 4. Methodology & Mathematical Frameworks

### 4.1 The "Polyphonic Score" (Normalized Shannon Diversity)
To measure whether a user is trapped in a mono-genre echo chamber or actively consuming a diverse musical palette, each user's streaming history was joined against multi-genre tag lists and exploded into individual `(user, track, genre)` triplets. 

For a user $u$, let $p_{u,i}$ represent the relative probability of observing genre $i$ across their total historical plays. The raw Shannon Entropy $H_u$ is computed as:

$$H_u = -\sum_{i=1}^{k} p_{u,i} \ln(p_{u,i})$$

To enable cross-user comparability on a bounded scale of $[0, 1]$, $H_u$ is normalized by the maximum possible entropy $\ln(K)$, where $K$ is the total number of distinct genres across the corpus:

$$\text{Polyphonic Score}_u = \frac{H_u}{\ln(K)}$$

* **$\text{Score} \rightarrow 0$:** Indicates strict mono-genre concentration.
* **$\text{Score} \rightarrow 1$:** Indicates a perfectly uniform, omnivorous distribution across all available genres.

### 4.2 Genre Co-occurrence Network (Graph Theory)
Using `NetworkX`, an undirected weighted graph $G = (V, E)$ was constructed where nodes $V$ represent distinct genres (scaled by marginal frequency) and edges $E$ represent pairwise co-occurrences across tracks.
* **Statistical Filtering:** Applied a minimum support threshold ($\ge 100$ joint occurrences) to prune spurious links and isolate genuine stylistic affinity.
* **Community Detection:** Applied a Fruchterman-Reingold force-directed spring layout paired with **Greedy Modularity Maximization**, achieving modularity scores between $0.35$ and $0.45$ (confirming statistically significant clustering well above random graph baselines).

---

## 5. Key Analytical Findings & Insights

### 5.1 Seven Decades of Sonic Evolution ($1955\text{--}2024$)
Longitudinal analysis of normalized audio features reveals a dramatic structural shift in music production over the last 70 years:

| Audio Dimension | Historical Baseline ($1955\text{--}1964$) | Modern Era ($2020\text{--}2024$) | Relative Shift | Analytical Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Danceability** | $\sim 0.05$ | $1.00$ | **$+1,900\%$** | Continuous decadal rise toward groove-centric, rhythm-heavy production. |
| **Energy** | $\sim 0.14$ ($1965\text{--}69$) | $\sim 0.59$ (Peak: $0.83$) | **$+493\%$ (to peak)** | Surged through the early $2000\text{s}$ loudness war before moderating by $-29\%$. |
| **Acousticness** | $1.00$ | $\sim 0.14\text{--}0.23$ | **$-99\%$ (to trough)** | Massive replacement of organic instrumentation with synthesized/electronic textures. |
| **Valence (Mood)** | $1.00$ | $\sim 0.18$ | **$-82\%$** | Modern popular music is significantly more melancholic or emotionally ambivalent. |
| **Instrumentalness**| $1.00$ ($1960\text{--}64$) | $\sim 0.17$ | **$-83\%$** | Near-total dominance of vocal-driven track structures. |

### 5.2 Sonic Homogenization Across Global Genres
Radar profiling of modern top-performing genres—**Afrobeats, K-Pop, and Mainstream Pop**—demonstrates nearly overlapping acoustic signatures (high danceability $\approx 0.68$, high energy $\approx 0.67$, high loudness $\approx -6.1\text{ dB}$, and low acousticness). Despite originating from distinct cultural spheres (West Africa, East Asia, Western Pop), global hits converge onto an engineered, beat-driven production formula to satisfy algorithmic distribution.

Conversely, contrasting **Popularity Extremes** shows that low-popularity tracks ($\text{Popularity} \le 5$, $n = 13,929$) preserve regional and acoustic diversity (led by *Iranian* at $5.5\%$, *Romance* at $4.7\%$, *Country* at $4.4\%$, and *Jazz* at $2.7\%$), while high-popularity tracks ($\text{Popularity} \ge 80$, $n = 833$) consolidate heavily around *Pop* ($8.5\%$), *Dance* ($4.9\%$), *Rock* ($4.2\%$), and *Hip-Hop* ($4.1\%$).

### 5.3 Cross-Platform Virality vs. Global Chart Rank
Correlation analysis between multi-platform metrics and **Spotify Global Chart Rank (`adjusted_rank`)** challenges the assumption that social media virality automatically guarantees chart dominance:
* **Spotify Playlist Reach ($r \approx +0.50$):** The single strongest predictor of global chart rank. Platform-curated playlist placement is the primary gatekeeper of hit status.
* **Spotify Internal Popularity Score ($r \approx +0.27$):** Outperforms raw cumulative stream counts ($r \approx -0.14$).
* **TikTok Engagement ($r \approx +0.15\text{ to }+0.17$):** TikTok views ($r = +0.15$), likes ($r = +0.17$), and post volume ($r = +0.12$) show only modest positive correlation with sustained chart ranking, indicating that short-form viral spikes often fail to convert into full-track retention.
* **YouTube Views & Likes ($r \approx -0.20, -0.18$):** Exhibit weak negative correlation with current Spotify weekly rank, reflecting platform audience divergence and catalog lag.

### 5.4 Micro-Level User Archetypes & Telemetry
Evaluating the 5 deep-archive cohorts revealed distinct behavioral archetypes across skip behavior, active search queries, chart overlap, and diversity:

| User Cohort | Polyphonic Score (Diversity) | Global Chart Overlap | Skip Rate ($<30\text{ s}$) | Avg. Skip Time | Active Search Queries | Behavioral Archetype |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SWC1`** | **$0.69$** (Highest) | $5.5\%$ ($14\%$ Top-50 Track) | $16.4\%$ | $\sim 4.4\text{ s}$ | **$5,937$** | **Active Omnivore:** High agency, heavy direct search, broad genre palette. |
| **`AND0`** | **$0.68$** | **$8.7\%$** ($12\%$ Top-50 Artist) | $21.2\%$ | $\sim 7.0\text{ s}$ | $374$ | **Mainstream Explorer:** Follows chart artists via playlists but explores their deep cuts. |
| **`ANR0`** | $0.57$ | $3.3\%$ | $37.9\%$ | $\sim 3.0\text{ s}$ | $17$ | **Passive Niche Listener:** Relies almost entirely on algorithmic feeds outside the mainstream. |
| **`SWC0`** | $0.56$ | $3.9\%$ | **$70.0\%$** | **$\sim 2.8\text{ s}$** | N/A | **Rapid Filterer:** Skips $70\%$ of tracks within $<3\text{ seconds}$ to curate mood. |
| **`ANS0`** | **$0.54$** (Lowest) | **$3.1\%$** | **$5.2\%$** | **$\sim 8.8\text{ s}$** | N/A | **Deep Specialist:** Narrow genre focus, lowest chart overlap, rarely skips ($94.8\%$ completion). |

* **The Skip-Time Inverse Law:** A strong inverse relationship exists between skip frequency and decision time. High-skip users (`SWC0` at $70\%$) abandon tracks almost instantaneously ($\sim 2.8\text{ s}$), whereas low-skip users (`ANS0` at $5.2\%$) invest nearly $9\text{ seconds}$ before deciding to skip.
* **Catalog Dominance (Release Recency):** Across *all* user cohorts, **$>80\%$ of distinct streamed tracks were released $>1\text{ year}$ prior to first listen**, while fresh releases ($\le 1\text{ week}$) accounted for $<1\%$.

### 5.5 Genre Co-occurrence Network Topology
The `NetworkX` community detection model uncovered three structural tiers in modern music consumption:
1. **The Central Supercluster:** A highly dense hub comprising *Pop*, *Dance Pop*, *Hip-Hop*, and *Urban Contemporary*, acting as the gravitational center of streaming traffic.
2. **Peripheral Stylistic Islands:** Tightly connected, self-contained cliques such as Metal (*Metalcore*, *Death-Metal*), Underground Electronic (*Minimal-Techno*, *Deep House*), and Regional Pop (*Mandopop*, *Cantopop*).
3. **High-Betweenness Bridge Genres:** Genres like **R&B** and **EDM** exhibit high betweenness centrality, functioning as critical structural bridges that connect peripheral subcultures to the mainstream supercluster.

---

## 6. Machine Learning: Multi-Class Genre Classification

To test whether modern genres can be predicted algorithmically, a two-stage supervised learning experiment was conducted across **32 consolidated target genres** (filtering for post-1970 tracks with $\ge 940$ instances).

### 6.1 Model Comparison & Feature Engineering Impact

| Model Architecture | Feature Set | Train / Val Split | Validation Accuracy | Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline: Random Forest** (`n_estimators=100`) | 11 raw Winsorized Spotify audio features only. | $80\% / 20\%$ (Stratified) | **$24.00\%$** (Macro F1: $0.12$) | Intrinsic audio features alone fail to separate modern genres due to sonic homogenization. |
| **Enhanced: XGBoost Classifier** (Tuned + Early Stopping) | Audio features + **`track_age`** + **Interaction Terms** + **Bucketized `popularity`**. | $85\% / 15\%$ (Stratified) | **$65.41\%$** ($+41.41\%$ gain) | Blending sonic properties with temporal and behavioral popularity indicators drastically boosts predictive power. |

### 6.2 Feature Engineering Pipeline
The $+41.41\%$ accuracy leap in the **XGBoost** model was driven by domain-specific feature engineering:
1. **Temporal Context (`track_age`):** Calculated from `release_date` to capture era-specific production norms.
2. **Domain Interaction Terms:**
   * $\text{dance\_energy} = \text{danceability} \times \text{energy}$ (captures club/festival intensity).
   * $\text{acoustic\_instrumental} = \text{acousticness} \times \text{instrumentalness}$ (isolates organic/ambient compositions).
   * $\text{tempo\_valence} = \text{tempo} \times \text{valence}$ (captures upbeat vs. slow-melancholic pacing).
3. **Behavioral Context (`popularity_bucket_encoded`):** Discretized popularity into *Low*, *Medium*, and *High* tiers to reflect mainstream vs. niche market positioning.

### 6.3 Out-of-Sample Deployment (73-Country Inference)
The trained XGBoost pipeline (including training-derived Winsorization bounds and standard scalers) was deployed on the unseen `TopByCountries` dataset to map global genre prevalence:
* **Global Dominance:** *Dance* ($106,065$ predictions) and *Chill* ($32,134$) lead global chart distributions, followed by *Anime* ($11,986$), *Deep-House* ($8,630$), and *Country* ($8,554$).
* **Regional Signatures:** Cross-country facet analysis highlighted distinct cultural fingerprints—such as the high prevalence of *Anime* and *Dance* in **Japan** and **South Korea**, *Chill* and *Anime* leading in **Egypt**, and *Dance*, *Country*, and *Alt* dominating the **United States** and **United Kingdom**.

---

## 7. Preserved Code Architecture (Python OOP Ingestion)

Below is the preserved Python object-oriented ingestion pipeline used to unpack, classify, and index multi-user Spotify Account and Extended Streaming ZIP archives into $O(1)$ lookup data structures:

```python
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

WORKDIR = Path(".")

# 1. Automated Idempotent Archive Extraction
for z in WORKDIR.glob("*.zip"):
    target = WORKDIR / z.stem
    if not target.exists():
        with ZipFile(z) as zf:
            zf.extractall(target)
        print(f"Extracted {z.name} -> {target}")

# 2. Data Class Object Model for Multi-Modal Telemetry
@dataclass
class AccountFiles:
    basic_stream: List[Path]
    playlists:    List[Path]
    profile:      List[Path]
    marketing:    List[Path]
    other:        List[Path] = field(default_factory=list)

@dataclass
class ExtendedStreamingFiles:
    audio: List[Path]
    video: List[Path]

@dataclass
class UserBundle:
    account:  AccountFiles
    extended: ExtendedStreamingFiles

# 3. Deterministic Path-Based Classifier & Bundle Builder
def build_bundle(uid: str) -> UserBundle:
    acc_root = WORKDIR / f"{uid}_account"
    ext_root = WORKDIR / f"{uid}_extendedstreaming"

    basic_stream = list(acc_root.glob("StreamingHistory_*"))
    playlists    = [
        *acc_root.glob("Playlist*.json"),
        *acc_root.glob("YourLibrary.json"),
        *acc_root.glob("YourSoundCapsule.json")
    ]
    profile      = [
        p for p in acc_root.glob("*.json")
        if p.stem in {"Identity", "Identifiers", "Userdata", "UserAddress"}
    ]
    marketing    = [
        p for p in acc_root.glob("*.json")
        if p.stem in {"Inferences", "Marquee"}
    ]

    accounted = {p.name for p in basic_stream + playlists + profile + marketing}
    other     = [p for p in acc_root.glob("*.json") if p.name not in accounted]

    account_files = AccountFiles(basic_stream, playlists, profile, marketing, other)
    audio         = list(ext_root.glob("Streaming_History_Audio_*.json"))
    video         = list(ext_root.glob("Streaming_History_Video_*.json"))
    ext_files     = ExtendedStreamingFiles(audio, video)

    return UserBundle(account=account_files, extended=ext_files)

# 4. O(1) In-Memory User Catalogue Construction
users: Dict[str, UserBundle] = {
    uid: build_bundle(uid) for uid in ["SWC0", "SWC1", "AND0", "ANR0", "ANS0"]
}
```

---

## 8. Product & Industry Recommendations

1. **Bridge-Genre Serendipity Injection:** Recommendation engines should leverage **high-betweenness bridge genres** (e.g., *R&B*, *EDM*) identified in the co-occurrence network to gradually transition users from the central Pop supercluster into peripheral genre islands without triggering immediate $<3\text{ s}$ skips.
2. **Polyphonic Score Personalization:** Segment users dynamically by their normalized Shannon Entropy. High-polyphonic users (`Score` $\ge 0.65$) should receive higher exploration weights in *Discover Weekly*, whereas low-polyphonic specialists (`Score` $\le 0.55$) benefit from deep-catalog cuts within their core genre.
3. **Catalog vs. Fresh Release Rebalancing:** Because $>80\%$ of distinct user streams come from tracks $>1\text{ year}$ old, platforms need dedicated "Fresh Cultural Discovery" carousels to prevent historical catalog lock-in.
4. **Algorithmic Fairness Audits:** Implement exposure-equity metrics to ensure regional genres (which currently cluster in the $\le 5$ popularity tier) receive fair playlist reach—the primary driver ($r = +0.50$) of chart visibility.

---

## 9. Repository Structure & Code Availability Note

```text
├── docs/
│   └── Polyphonic_Identities_Dissertation_Swapnil_Chhetri.pdf   # Full 71-page MSc Dissertation Report
├── src/
│   └── spotify_archive_loader.py                                # Reconstructed OOP ingestion module
└── README.md                                                    # Executive Summary & Technical Documentation
```

> **Note on Source Code Availability:** Due to an unexpected local hardware drive failure following the submission of this dissertation in May 2025, the original raw R preprocessing scripts and Jupyter notebooks containing the exploratory visualizations and XGBoost training loops were lost. 
> 
> To ensure this repository remains a transparent and valuable technical portfolio resource, the **complete 71-page dissertation PDF** (including all statistical tables, network graphs, confusion insights, and methodology logs) is hosted in this repository alongside the **reconstructed Python data-loading architecture** preserved in the dissertation figures.

---

## 10. Author & Contact

**Swapnil Chhetri**  
*MSc in Data Analytics — London Metropolitan University / Islington College*  

* **LinkedIn:** [Add Your LinkedIn URL Here]
* **Email:** [Add Your Email Address Here]
* **GitHub:** [Add Your GitHub Profile URL Here]
* **Location:** Open to Data Analyst / Data Scientist / Analytics Engineer roles.