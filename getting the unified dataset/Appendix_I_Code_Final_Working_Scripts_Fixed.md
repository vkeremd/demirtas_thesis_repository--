# Appendix I: Code - Final Working Scripts

## Overview
This appendix presents the final working code that successfully created the dataset `daily_tweets_with_population_and_protests_complete.csv`. I've included only the scripts that actually solved the problems, not the failed attempts.

## Phase 1: Population Data Fix - Final Working Solution

### 1.1 Ultimate Comprehensive Population Fix
**File:** `ultimate_comprehensive_fix.py`

This script provides the final, working solution for all population data issues encountered during the thesis analysis. It ensures that all population data comes from legitimate, verifiable sources and is properly formatted for analysis.

**Key Features:**
- Comprehensive data validation
- Multiple source verification
- Backup verification systems
- Complete documentation with verification links

**Usage:**
```bash
python ultimate_comprehensive_fix.py
```

### 1.2 Population Data Verification
**File:** `verify_population_data.py`

This script verifies that all population data comes from real, publicly available sources. It provides direct links to Census Bureau data for verification.

**Key Features:**
- Sample county verifications with direct links
- Data source explanations
- Verification methods documentation
- Public accessibility confirmation

**Usage:**
```bash
python verify_population_data.py
```

## Phase 2: Tweet and Protest Data Combination

### 2.1 Working Combination Script
**File:** `combine_tweets_and_protests_working.py`

This script successfully combines tweet data with protest event data from ACLED, handling all FIPS code format differences and creating the final analysis dataset.

**Key Features:**
- Robust CSV parsing for protest data
- Proper FIPS code handling
- Geographic aggregation by county and date
- Merge quality verification
- Final dataset creation

**Usage:**
```bash
python combine_tweets_and_protests_working.py
```

**Input Files:**
- `daily_tweets_with_population_final_ultimate.csv` - Tweet data with population
- `protests_acled_Data.csv` - ACLED protest event data

**Output File:**
- `daily_tweets_with_population_and_protests_complete.csv` - Final analysis dataset

## Phase 3: Data Quality Verification

### 3.1 Merge Quality Verification
The combination script includes built-in quality verification that checks:
- Missing data identification
- Data type validation
- Duplicate detection
- Distribution analysis for both tweet and protest data

### 3.2 Final Dataset Structure
The final dataset contains:
- **fips_code**: County identifier
- **county_name**: County name
- **state_name**: State abbreviation
- **date**: Date of observation
- **year, month, day, day_of_week**: Date components
- **injustice_label, is_injustice**: Tweet classification
- **tweet_count**: Total tweets
- **total_popE**: Population count
- **tweets_per_1000**: Per-capita tweet rate
- **injustice_tweets_per_1000**: Per-capita injustice tweet rate
- **non_injustice_tweets_per_1000**: Per-capita non-injustice tweet rate
- **protest_count**: Number of protests on that date

## Phase 4: Analysis Readiness

### 4.1 Dataset Summary Statistics
The final dataset provides:
- Total observations: Comprehensive county-date combinations
- Unique counties: Geographic coverage
- Date range: Temporal coverage
- Tweet and protest totals: Activity measures
- Counties with protests: Geographic distribution

### 4.2 Data Quality Assurance
All data has been verified for:
- **Source legitimacy**: All population data from US Census Bureau
- **Format consistency**: Proper FIPS codes and date formats
- **Completeness**: No missing critical data
- **Accuracy**: Cross-verified with multiple sources

## Implementation Notes

### Data Sources
- **Population Data**: US Census Bureau 2020 Census
- **Tweet Data**: Processed and classified tweets
- **Protest Data**: ACLED (Armed Conflict Location & Event Data Project)
- **Location Data**: Twitter place data mapped to FIPS codes

### Verification Process
1. Cross-reference with Census Bureau website
2. Check county government websites
3. Verify with state demographic offices
4. Document all sources with direct links

### Quality Checks
1. Population totals match official Census data
2. FIPS codes are correctly formatted
3. County names match official designations
4. State abbreviations are standard

## Conclusion

These scripts represent the final, working solution that successfully created the analysis dataset. All data has been verified, validated, and is ready for econometric analysis. The combination of tweet discourse data with protest event data provides a unique dataset for analyzing the relationship between social media discourse and collective action during the Black Lives Matter movement.
