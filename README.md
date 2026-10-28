# Conspiracy Leaders: A Dataset of Behavioral and Psycholinguistic Precursors to Conspiracy Community Founding
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

