# From Hashtags to the Streets: Police Killings, Perceived-Injustice Discourse, and Protest Dynamics via a Novel Transformer-Based BERTweet Fine-Tuned Model Approach

This repository contains the complete code and analysis for my Master's Thesis in Economics at LMU Munich. The thesis examines the relationship between social media discourse about racial injustice and protest activity during the Black Lives Matter movement. The analysis employs a novel BERTweet-based approach to classify injustice-related content and links this discourse to protest events using event study methodology.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Data](#data)
- [Running the Code](#running-the-code)
  - [Phase 1: Complete BERTweet Workflow](#phase-1-complete-bertweet-workflow)
  - [Phase 2: Getting the Unified Dataset](#phase-2-getting-the-unified-dataset)
  - [Phase 3: Running the Empirical Model](#phase-3-running-the-empirical-model)
- [Reproducibility](#reproducibility)
- [Contact](#contact)

## Overview 

This thesis investigates the relationship between social media discourse about racial injustice and protest activity during the Black Lives Matter movement. The research employs a sophisticated machine learning pipeline using BERTweet models to classify injustice-related content in millions of tweets, then links this discourse to protest events using event study methodology.

The project is structured to allow for a step-by-step reproduction of the analysis, from raw tweet processing and model training to final econometric analysis and result generation.

### Key Research Questions:
1. **How does social media discourse about racial injustice relate to protest activity?**
2. **What are the dynamic effects of discourse intensity on protest mobilization?**
3. **How do these relationships vary across counties with different characteristics?**


## Repository Structure

```bash
/Complete BERTweet Work Flow/           # Phase 1: Model Training & Classification
  ├── Complete_BERTweet_Workflow_Documentation.py  # Complete workflow (Appendix I)
  ├── Appendix_II_Log_File_Example.txt             # Execution log 
  └── interpret_dose_binning_results.py            # Sensitivity analysis

/getting the unified dataset/           # Phase 2: Data Integration
  ├── combine_tweets_and_protests_working_fixed.py # Tweet-protest combination
  ├── ultimate_comprehensive_fix_fixed.py          # Population data fix
  ├── verify_population_data_fixed.py              # Data verification
  ├── Appendix_I_Code_Final_Working_Scripts_Fixed.md  # Code documentation
  └── Appendix_II_Log_File_Final_Working_Fixed.md     # Execution log

/empirical_model/                       # Phase 3: Econometric Analysis
  ├── empirical_equation_code.py                    # Model specification
  ├── empirical_equation_implementation.py          # Implementation
  ├── empiricallyverified_ppml_analysis.py          # PPML estimation
  ├── comprehensive_thesis_analysis.py              # Main analysis
  ├── comprehensive_dynamic_did_analysis.py         # Dynamic effects
  ├── comprehensive_robustness_analysis.py          # Robustness checks
  └── REAL_analysis_no_fake_data.py                # Final analysis

/data/                                  # Data files (Git LFS)
  ├── daily_tweets_with_population_and_protests_complete.csv  # ★ Final dataset
  ├── all_tweets_classified_20250627_120357.csv    # Classified tweets
  ├── protests_acled_Data.csv                       # ACLED protest data
  └── acscounty_level_controls.csv                  # Census controls

/models/                                 # Trained models
  ├── bertweet_injustice_final/                     # Injustice classifier
  └── bertweet_systemic_racism_improved_final/      # Systemic racism classifier

/results/                               # Analysis outputs
  ├── *.png, *.pdf                     # Visualizations
  ├── *.csv                            # Results tables
  └── *.json                           # Model outputs
```

## Setup & Dependencies

Before running the code, ensure you have the following environment setup:

### **Python Version**: Python 3.9 or higher

### **Required Packages**:
```bash
# Core data science
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Deep learning & NLP
torch>=2.0.0
transformers>=4.40.0
datasets>=2.0.0

# Econometrics
statsmodels>=0.13.0
linearmodels>=4.25.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Geographic data
geopandas>=0.12.0
shapely>=1.8.0

# Utilities
tqdm>=4.64.0
requests>=2.28.0
```

### **Installation**:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Data

The repository uses several data sources:

### **Primary Data Sources**:
1. **Tweet Data** - Raw JSON files (proprietary, not included)
2. **ACLED Protest Data** - Armed Conflict Location & Event Data Project
3. **Census Population Data** - U.S. Census Bureau county-level data
4. **Location Mapping** - Twitter place data to FIPS county codes

### **Processed Datasets**:
- **`daily_tweets_with_population_and_protests_complete.csv`** - Final analysis dataset
- **`all_tweets_classified_20250627_120357.csv`** - All classified tweets
- **`protests_acled_Data.csv`** - ACLED protest event data

### **Data Access**:
Large datasets are managed with **Git LFS** to keep the repository size manageable. See the [Data Sources](#data) section for access instructions.

## Running the Code

The analysis follows a three-phase pipeline. Each phase builds upon the previous one and produces outputs consumed by the next stage.

### Phase 1: Complete BERTweet Workflow

**Purpose**: Train BERTweet models to classify injustice-related content in tweets.

**Location**: `/Complete BERTweet Work Flow/`

```bash
# Navigate to the BERTweet workflow directory
cd "Complete BERTweet Work Flow"

# Run the complete workflow
python Complete_BERTweet_Workflow_Documentation.py
```

**What this does**:
1. **Raw Data Processing**: Extracts location information from tweet JSON files
2. **Training Data Preparation**: Creates labeled dataset for model training
3. **BERTweet Model Training**: Fine-tunes models for injustice classification
4. **Tweet Classification**: Applies trained models to classify all tweets
5. **Geographic Aggregation**: Creates daily county-level tweet counts
6. **Population Integration**: Adds population data for per-capita analysis

**Outputs**:
- Trained BERTweet models in `/models/`
- Classified tweet data
- Daily tweet counts by county

### Phase 2: Getting the Unified Dataset

**Purpose**: Combine tweet data with protest event data to create the final analysis dataset.

**Location**: `/getting the unified dataset/`

```bash
# Navigate to the unified dataset directory
cd "getting the unified dataset"

# Run the data combination workflow
python combine_tweets_and_protests_working_fixed.py

# Verify population data sources
python verify_population_data_fixed.py

# Apply comprehensive population fixes
python ultimate_comprehensive_fix_fixed.py
```

**What this does**:
1. **Tweet-Protest Combination**: Merges classified tweets with ACLED protest data
2. **Population Data Verification**: Ensures all population data comes from legitimate sources
3. **Data Quality Checks**: Validates merge quality and data integrity
4. **Final Dataset Creation**: Produces the unified analysis dataset

**Outputs**:
- **`daily_tweets_with_population_and_protests_complete.csv`** - Final analysis dataset
- Data quality verification reports
- Population data verification documentation

### Phase 3: Running the Empirical Model

**Purpose**: Conduct econometric analysis to test the relationship between discourse and protest activity.

**Location**: `/empirical_model/` (files in root directory)

```bash
# Navigate back to root directory
cd ..

# Run the main econometric analysis
python comprehensive_thesis_analysis.py

# Run dynamic treatment effects analysis
python comprehensive_dynamic_did_analysis.py

# Run robustness checks
python comprehensive_robustness_analysis.py

# Run the final analysis with real data
python REAL_analysis_no_fake_data.py
```

**What this does**:
1. **Event Study Analysis**: Tests causal effects using natural experiments
2. **Dynamic Treatment Effects**: Examines time-varying effects
3. **Robustness Checks**: Validates findings across specifications
4. **Heterogeneity Analysis**: Explores variation across subgroups
5. **Visualization**: Creates figures and tables for presentation

**Outputs**:
- Regression results and tables
- Dynamic treatment effect plots
- Robustness check results
- Final thesis figures and visualizations

## Detailed Workflow

### **Step-by-Step Execution**:

1. **Start with Phase 1**: Train BERTweet models and classify tweets
2. **Proceed to Phase 2**: Combine data and create unified dataset
3. **Complete with Phase 3**: Run econometric analysis and generate results

### **Key Files for Each Phase**:

#### **Phase 1 - BERTweet Workflow**:
- `Complete_BERTweet_Workflow_Documentation.py` - Complete pipeline
- `train_model.py` - Model training
- `classify_all_tweets.py` - Tweet classification

#### **Phase 2 - Unified Dataset**:
- `combine_tweets_and_protests_working_fixed.py` - Data combination
- `verify_population_data_fixed.py` - Data verification
- `ultimate_comprehensive_fix_fixed.py` - Population fixes

#### **Phase 3 - Empirical Model**:
- `comprehensive_thesis_analysis.py` - Main analysis
- `comprehensive_dynamic_did_analysis.py` - Dynamic effects
- `empiricallyverified_ppml_analysis.py` - PPML estimation

### **Expected Outputs**:

#### **Phase 1 Outputs**:
- Trained BERTweet models
- Classified tweet dataset
- Daily tweet counts by county

#### **Phase 2 Outputs**:
- Unified analysis dataset
- Data quality verification
- Population data documentation

#### **Phase 3 Outputs**:
- Regression results
- Dynamic treatment effects
- Robustness check results
- Final visualizations

## Reproducibility 

To replicate or extend this study, please ensure the following:

### **Data Requirements**:
- Access to raw tweet data (requires separate authorization)
- ACLED protest data (publicly available)
- Census population data (publicly available)

### **Code Execution**:
1. **Follow the phase order**: Execute phases 1, 2, and 3 in sequence
2. **Use the same data processing steps** as detailed in each script
3. **Document any deviations** from the original analysis
4. **Verify data sources** using the provided verification scripts

### **Quality Assurance**:
- All population data is verified against US Census Bureau sources
- Model performance metrics are documented
- Robustness checks are included for all specifications
- Complete documentation is provided for each step

### **Model Artifacts**:
- Trained BERTweet models are available for download
- Training data and validation results are included
- Model performance metrics are documented

## Contact

For any questions or further information, please contact:

| Name                    | Email                                    | Institution           |
| ----------------------- | ---------------------------------------- | --------------------- |
| Vedat Kerem Demirtas    | [k.demirtas@campus.lmu.de](mailto:k.demirtas@campus.lmu.de) | LMU Munich            |

### **Feedback & Collaboration**:
- **Issues**: Report bugs or problems via GitHub Issues
- **Questions**: Email for academic inquiries
- **Collaboration**: Open to research partnerships
- **Citations**: Please cite this work if used in your research

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@thesis{demirtas2024voices,
  title={# From Hashtags to the Streets: Police Killings, Perceived-Injustice Discourse, and Protest Dynamics via a Novel Transformer-Based BERTweet Fine-Tuned Model Approach
},
  author={Demirtas, Vedat Kerem},
  year={2025},
  institution={Ludwig Maximilian University of Munich},
  type={Master's Thesis}
}
```

---

## Security & Ethics

- **No API keys or credentials** included in the code
- **Public datasets only** - no proprietary data shared
- **Privacy compliant** - individual tweets not identifiable
- **Academic use** - research purposes only

---

*This repository represents the complete computational pipeline for analyzing the relationship between social media discourse about racial injustice and protest activity during the Black Lives Matter movement. The BERTweet-based approach provides a sophisticated method for measuring discourse intensity that captures the nuanced language patterns of social media conversations about systemic racism and police violence.*
