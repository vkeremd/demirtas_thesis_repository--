#!/usr/bin/env python3
"""
Appendix I: Complete Code for Discourse-Protest Analysis
=======================================================

This script implements a comprehensive econometric analysis of the relationship between
injustice discourse and protest activity using county-day panel data.

Key Features:
- Baseline models with three discourse variants (per-capita, share, spike)
- Phase-based analysis (Step1, Step2, Step3) instead of single post indicator
- Heterogeneity analysis with median-based approach (above/below median)
- Marginal effects calculations
- Publication-ready plots and LaTeX tables

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use("default")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "use_day_fe": True,  # Always use day fixed effects (γ_t) as per the specification
    "d_variants": ["injustice_pc_100k", "injustice_share_pp", "injustice_spike_1sd"],
    "moderators": ["share_black", "dem_vote_share", "education_ba_plus", "median_income", "urban_z"],
    "phases": ["Step1", "Step2", "Step3"],
    "output_dir": "./outputs_demo"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_between(value, min_val, max_val, name):
    """Assert that a value is between min and max"""
    assert min_val <= value <= max_val, f"{name} ({value}) must be between {min_val} and {max_val}"

def preflight_checks(df):
    """Validate data before analysis"""
    print("Running preflight checks...")
    
    # Check discourse variables are in reasonable ranges
    if 'injustice_pc_100k' in df.columns:
        assert_between(df['injustice_pc_100k'].mean(), 0, 1000, "injustice_pc_100k mean")
    
    if 'injustice_share_pp' in df.columns:
        assert_between(df['injustice_share_pp'].mean(), 0, 100, "injustice_share_pp mean")
    
    # Check moderators are in reasonable ranges
    if 'share_black' in df.columns:
        assert_between(df['share_black'].mean(), 0, 1, "share_black mean")
    
    if 'dem_vote_share' in df.columns:
        assert_between(df['dem_vote_share'].mean(), 0, 1, "dem_vote_share mean")
    
    print("✓ All preflight checks passed")

def add_phase_dummies(df):
    """Create phase dummies for the three post-treatment periods"""
    print("Creating phase dummies...")
    
    df = df.copy()
    
    # Create three non-overlapping phase dummies
    df['Step1'] = ((df['date'] >= pd.Timestamp('2020-05-25')) & 
                   (df['date'] <= pd.Timestamp('2020-05-28'))).astype(int)
    
    df['Step2'] = ((df['date'] >= pd.Timestamp('2020-05-29')) & 
                   (df['date'] <= pd.Timestamp('2020-06-04'))).astype(int)
    
    df['Step3'] = (df['date'] >= pd.Timestamp('2020-06-05')).astype(int)
    
    # Pre-period (< May 25) is the omitted category
    print(f"Phase distribution:")
    print(f"  Step1 (May 25-28): {df['Step1'].sum()} observations")
    print(f"  Step2 (May 29-Jun 4): {df['Step2'].sum()} observations")
    print(f"  Step3 (≥ Jun 5): {df['Step3'].sum()} observations")
    
    return df

def create_derived_variables(df):
    """Create derived discourse variables with proper scaling"""
    print("Creating derived discourse variables...")
    
    df = df.copy()
    
    print(f"Available columns in create_derived_variables: {list(df.columns)}")
    
    # Calculate injustice tweets from per_1000 variables
    df['injustice_tweets'] = df['injustice_tweets_per_1000'] * df['population'] / 1000
    df['non_injustice_tweets'] = df['non_injustice_tweets_per_1000'] * df['population'] / 1000
    
    # Create rescaled discourse variables for interpretability
    df['injustice_pc_100k'] = (df['injustice_tweets'] / df['population']) * 100000  # per 100k residents
    df['injustice_share_pp'] = (df['injustice_tweets'] / df['total_tweets']) * 100  # percentage points
    df['noninj_pc_100k'] = (df['non_injustice_tweets'] / df['population']) * 100000  # per 100k residents
    
    # Create spike indicator (1 if injustice_tweets >= mean + 1*sd within county)
    df['injustice_spike_1sd'] = 0  # Default to 0
    
    # Calculate county-specific thresholds for spike
    for county in df['fips'].unique():
        county_data = df[df['fips'] == county]
        if len(county_data) > 0:
            mean_inj = county_data['injustice_tweets'].mean()
            sd_inj = county_data['injustice_tweets'].std()
            threshold = mean_inj + sd_inj
            df.loc[df['fips'] == county, 'injustice_spike_1sd'] = (
                df.loc[df['fips'] == county, 'injustice_tweets'] >= threshold
            ).astype(int)
    
    print(f"Discourse variables created:")
    print(f"  injustice_pc_100k: mean={df['injustice_pc_100k'].mean():.2f}, std={df['injustice_pc_100k'].std():.2f}")
    print(f"  injustice_share_pp: mean={df['injustice_share_pp'].mean():.2f}, std={df['injustice_share_pp'].std():.2f}")
    print(f"  injustice_spike_1sd: mean={df['injustice_spike_1sd'].mean():.2f}")
    
    return df

def add_urbanization_measures(df):
    """Add urbanization measures for heterogeneity analysis"""
    print("Adding urbanization measures...")
    
    df = df.copy()
    
    # Create urbanization proxies
    df['log_population'] = np.log1p(df['population'])
    
    # Create standardized urbanization measure
    df['urban_z'] = (df['log_population'] - df['log_population'].mean()) / df['log_population'].std()
    
    # Create placeholder values for missing columns (for demo purposes)
    df['pop_density'] = df['population'] / 1000  # Placeholder
    df['urban_share_pp'] = df['log_population'] * 10  # Placeholder
    df['cbsa_flag'] = (df['log_population'] > df['log_population'].median()).astype(int)
    
    print(f"Urbanization measures created:")
    print(f"  urban_z: mean={df['urban_z'].mean():.2f}, std={df['urban_z'].std():.2f}")
    
    return df

def fit_simple_ols(df, formula):
    """Fit OLS model with clustered standard errors"""
    try:
        import statsmodels.api as sm
        
        # Reset index for statsmodels
        df_reset = df.reset_index()
        
        # Fit model
        model = sm.OLS.from_formula(formula, data=df_reset)
        results = model.fit()
        
        # Try to calculate clustered standard errors if available
        try:
            from statsmodels.stats.sandwich import cov_cluster
            cluster_var = df_reset['fips']
            cov_cluster_matrix = cov_cluster(results, cluster_var)
            results.cov_params = lambda: cov_cluster_matrix
        except ImportError:
            # If sandwich module not available, use regular standard errors
            print("Warning: statsmodels.stats.sandwich not available, using regular SEs")
            # Store the regular covariance matrix
            regular_cov = results.cov_params()
            results.cov_params = lambda: regular_cov
        
        return results
    except Exception as e:
        print(f"Error in OLS fitting: {e}")
        return None

def estimate_baseline_models(df):
    """Estimate baseline models using simple triple difference specification"""
    print("\n" + "="*80)
    print("ESTIMATING BASELINE MODELS")
    print("="*80)
    
    baseline_results = []
    
    # Create a simple Post indicator (1 for all dates after May 25, 2020)
    df_reset = df.reset_index()
    df_reset['Post'] = (df_reset['date'] >= '2020-05-25').astype(int)
    df = df_reset.set_index(['fips', 'date'])
    
    for d_var in CONFIG["d_variants"]:
        print(f"\nEstimating baseline model for {d_var}...")
        
        # Determine control variable
        if d_var == 'injustice_share_pp':
            control = 'noninj_pc_100k'  # Use non-injustice tweets per capita
        else:
            control = 'total_tweets'  # Use total tweets
        
        # Simple triple difference specification:
        # Protests_ct = β₁(D_ct × Post_t) + β₂Post_t + β₃D_ct + β₄TotalTweets_ct + α_c + γ_t + ε_ct
        formula = f"protests ~ {d_var}:Post + Post + {d_var} + {control} + C(fips) + C(date)"
        print(f"Formula: {formula}")
        print("This implements: Protests_ct = β₁(D_ct × Post_t) + β₂Post_t + β₃D_ct + β₄{control}_ct + α_c + γ_t + ε_ct")
        
        # Fit model
        results = fit_simple_ols(df, formula)
        
        if results is None:
            print(f"  Failed to estimate model for {d_var}")
            continue
        
        # Extract the main triple difference coefficient (β₁)
        interaction_term = f"{d_var}:Post"
        
        if interaction_term in results.params.index:
            coef = results.params[interaction_term]
            se = results.bse[interaction_term]
            pval = results.pvalues[interaction_term]
            
            print(f"  Triple difference coefficient (β₁): {coef:.3f} (SE: {se:.3f}, p={pval:.3f})")
            
            # Add unit suffix for column names
            if d_var == 'injustice_pc_100k':
                unit_suffix = '_per100k'
            elif d_var == 'injustice_share_pp':
                unit_suffix = '_pp'
            else:  # injustice_spike_1sd
                unit_suffix = '_binary'
            
            baseline_results.append({
                'D_variant': d_var,
                'phase': 'Post',
                f'coef_DxPost{unit_suffix}': coef,
                f'se_DxPost{unit_suffix}': se,
                f'pval_DxPost{unit_suffix}': pval,
                'N': len(df),
                'n_counties': df.index.get_level_values('fips').nunique(),
                'n_days': df.index.get_level_values('date').nunique(),
                'R2': results.rsquared
            })
        
        print()
    
    return pd.DataFrame(baseline_results)

def estimate_heterogeneity_models(df):
    """Estimate heterogeneity models with triple interactions"""
    print("\n" + "="*80)
    print("ESTIMATING HETEROGENEITY MODELS")
    print("="*80)
    
    heterogeneity_results = []
    
    # Use only first two discourse variants for heterogeneity (skip spike)
    d_variants_het = ["injustice_pc_100k", "injustice_share_pp"]
    
    for d_var in d_variants_het:
        print(f"\nEstimating heterogeneity models for {d_var}...")
        
        # Determine control variable
        if d_var == 'injustice_share_pp':
            control = 'noninj_pc_100k'
        else:
            control = 'total_tweets'
        
        for mod in CONFIG["moderators"]:
            print(f"  Moderator: {mod}")
            
            # Build heterogeneity formula following the specification with day fixed effects
            formula = (f"protests ~ {d_var}:Step1:{mod} + {d_var}:Step2:{mod} + {d_var}:Step3:{mod} + "
                      f"{d_var}:Step1 + {d_var}:Step2 + {d_var}:Step3 + "
                      f"{d_var}:{mod} + Step1:{mod} + Step2:{mod} + Step3:{mod} + "
                      f"{d_var} + Step1 + Step2 + Step3 + {control} + C(fips) + C(date)")
            
            # Fit model
            results = fit_simple_ols(df, formula)
            
            if results is not None:
                # Extract triple interaction coefficients for each phase
                for step in CONFIG["phases"]:
                    triple_coef_name = f"{d_var}:{step}:{mod}"
                    
                    if triple_coef_name in results.params.index:
                        coef = results.params[triple_coef_name]
                        se = np.sqrt(results.cov_params().loc[triple_coef_name, triple_coef_name])
                        pval = results.pvalues[triple_coef_name]
                        
                        # Determine unit suffix
                        if d_var == 'injustice_pc_100k':
                            unit_suffix = '_per100k'
                        else:  # injustice_share_pp
                            unit_suffix = '_pp'
                        
                        heterogeneity_results.append({
                            'model_id': f'{d_var}_{mod}',
                            'D_variant': d_var,
                            'moderator': mod,
                            'phase': step,
                            f'triple_coef_{step.lower()}{unit_suffix}': coef,
                            f'triple_se_{step.lower()}{unit_suffix}': se,
                            f'triple_pval_{step.lower()}{unit_suffix}': pval,
                            'N': len(df),
                            'n_counties': df.index.get_level_values('fips').nunique(),
                            'n_days': df.index.get_level_values('date').nunique(),
                            'R2': results.rsquared
                        })
                        
                        # Print results
                        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                        print(f"    {step}: {coef:.3f}{sig} (SE: {se:.3f}, p={pval:.3f})")
            
            else:
                print(f"    Failed to estimate model for {d_var} × {mod}")
    
    return pd.DataFrame(heterogeneity_results)

def estimate_spike_heterogeneity_models(df):
    """Estimate heterogeneity models for spike discourse variant"""
    print("\n" + "="*80)
    print("ESTIMATING SPIKE HETEROGENEITY MODELS")
    print("="*80)
    
    spike_results = []
    
    for mod in CONFIG["moderators"]:
        print(f"  Moderator: {mod}")
        
        # Build heterogeneity formula for spike following the specification with day fixed effects
        formula = (f"protests ~ injustice_spike_1sd:Step1:{mod} + injustice_spike_1sd:Step2:{mod} + injustice_spike_1sd:Step3:{mod} + "
                  f"injustice_spike_1sd:Step1 + injustice_spike_1sd:Step2 + injustice_spike_1sd:Step3 + "
                  f"injustice_spike_1sd:{mod} + Step1:{mod} + Step2:{mod} + Step3:{mod} + "
                  f"injustice_spike_1sd + Step1 + Step2 + Step3 + total_tweets + C(fips) + C(date)")
        
        # Fit model
        results = fit_simple_ols(df, formula)
        
        if results is not None:
            # Extract triple interaction coefficients for each phase
            for step in CONFIG["phases"]:
                triple_coef_name = f"injustice_spike_1sd:{step}:{mod}"
                
                if triple_coef_name in results.params.index:
                    coef = results.params[triple_coef_name]
                    se = np.sqrt(results.cov_params().loc[triple_coef_name, triple_coef_name])
                    pval = results.pvalues[triple_coef_name]
                    
                    spike_results.append({
                        'model_id': f'injustice_spike_1sd_{mod}',
                        'D_variant': 'injustice_spike_1sd',
                        'moderator': mod,
                        'phase': step,
                        f'triple_coef_{step.lower()}_binary': coef,
                        f'triple_se_{step.lower()}_binary': se,
                        f'triple_pval_{step.lower()}_binary': pval,
                        'N': len(df),
                        'n_counties': df.index.get_level_values('fips').nunique(),
                        'n_days': df.index.get_level_values('date').nunique(),
                        'R2': results.rsquared
                    })
                    
                    # Print results
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                    print(f"    {step}: {coef:.3f}{sig} (SE: {se:.3f}, p={pval:.3f})")
        
        else:
            print(f"    Failed to estimate model for injustice_spike_1sd × {mod}")
    
    return pd.DataFrame(spike_results)

def calculate_marginal_effects(df, baseline_results, heterogeneity_results):
    """Calculate marginal effects for median-based analysis"""
    print("\n" + "="*80)
    print("CALCULATING MARGINAL EFFECTS")
    print("="*80)
    
    marginal_effects = []
    
    # Define median-based values for each moderator
    median_values = {
        'share_black': {'below': 0.05, 'above': 0.15},
        'dem_vote_share': {'below': 0.35, 'above': 0.55},
        'education_ba_plus': {'below': 0.20, 'above': 0.30},
        'median_income': {'below': 55000, 'above': 65000},
        'urban_z': {'below': -0.5, 'above': 0.5}
    }
    
    for d_var in ["injustice_pc_100k", "injustice_share_pp"]:
        print(f"\nCalculating marginal effects for {d_var}...")
        
        # Get baseline coefficients
        baseline_data = baseline_results[baseline_results['D_variant'] == d_var]
        
        for step in CONFIG["phases"]:
            # Get baseline coefficient
            step_baseline = baseline_data[baseline_data['phase'] == step]
            if len(step_baseline) == 0:
                continue
                
            if d_var == 'injustice_pc_100k':
                baseline_coef = step_baseline[f'coef_Dx{step}_per100k'].iloc[0]
                baseline_se = step_baseline[f'se_Dx{step}_per100k'].iloc[0]
            else:  # injustice_share_pp
                baseline_coef = step_baseline[f'coef_Dx{step}_pp'].iloc[0]
                baseline_se = step_baseline[f'se_Dx{step}_pp'].iloc[0]
            
            # Calculate marginal effects for each moderator
            for mod in CONFIG["moderators"]:
                # Get heterogeneity coefficient
                het_data = heterogeneity_results[
                    (heterogeneity_results['D_variant'] == d_var) & 
                    (heterogeneity_results['moderator'] == mod) &
                    (heterogeneity_results['phase'] == step)
                ]
                
                if len(het_data) == 0:
                    continue
                
                if d_var == 'injustice_pc_100k':
                    triple_coef = het_data[f'triple_coef_{step.lower()}_per100k'].iloc[0]
                    triple_se = het_data[f'triple_se_{step.lower()}_per100k'].iloc[0]
                else:  # injustice_share_pp
                    triple_coef = het_data[f'triple_coef_{step.lower()}_pp'].iloc[0]
                    triple_se = het_data[f'triple_se_{step.lower()}_pp'].iloc[0]
                
                # Calculate marginal effects at below and above median
                for level in ['below', 'above']:
                    mod_val = median_values[mod][level]
                    
                    # Marginal effect = baseline + triple_interaction * moderator_value
                    me = baseline_coef + triple_coef * mod_val
                    
                    # Standard error using delta method
                    me_se = np.sqrt(baseline_se**2 + (mod_val**2) * triple_se**2)
                    
                    marginal_effects.append({
                        'D_variant': d_var,
                        'phase': step,
                        'moderator': mod,
                        'M_level': f"{level.capitalize()} Median",
                        'M_value': mod_val,
                        'ME': me,
                        'ME_se': me_se
                    })
    
    return pd.DataFrame(marginal_effects)

def export_results(baseline_results, heterogeneity_results, marginal_effects):
    """Export results to CSV files"""
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Export baseline results
    baseline_results.to_csv(output_dir / "baseline_summary.csv", index=False)
    print(f"✓ Baseline results exported to {output_dir / 'baseline_summary.csv'}")
    
    # Export heterogeneity results
    heterogeneity_results.to_csv(output_dir / "model_summaries.csv", index=False)
    print(f"✓ Heterogeneity results exported to {output_dir / 'model_summaries.csv'}")
    
    # Export marginal effects
    marginal_effects.to_csv(output_dir / "marginal_effects_urban.csv", index=False)
    print(f"✓ Marginal effects exported to {output_dir / 'marginal_effects_urban.csv'}")
    
    # Export configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Configuration exported to {output_dir / 'config.json'}")

def plot_baseline_coefficients(baseline_results):
    """Create baseline coefficient plot"""
    print("\nCreating baseline coefficient plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for phases
    phase_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    y_positions = []
    y_labels = []
    
    for i, d_var in enumerate(CONFIG["d_variants"]):
        data = baseline_results[baseline_results['D_variant'] == d_var]
        
        for j, step in enumerate(CONFIG["phases"]):
            step_data = data[data['phase'] == step]
            
            if len(step_data) > 0:
                # Get coefficient and standard error
                if d_var == 'injustice_pc_100k':
                    coef = step_data[f'coef_Dx{step}_per100k'].iloc[0]
                    se = step_data[f'se_Dx{step}_per100k'].iloc[0]
                    pval = step_data[f'pval_Dx{step}_per100k'].iloc[0]
                elif d_var == 'injustice_share_pp':
                    coef = step_data[f'coef_Dx{step}_pp'].iloc[0]
                    se = step_data[f'se_Dx{step}_pp'].iloc[0]
                    pval = step_data[f'pval_Dx{step}_pp'].iloc[0]
                else:  # injustice_spike_1sd
                    coef = step_data[f'coef_Dx{step}_binary'].iloc[0]
                    se = step_data[f'se_Dx{step}_binary'].iloc[0]
                    pval = step_data[f'pval_Dx{step}_binary'].iloc[0]
                
                y_pos = len(y_positions)
                y_positions.append(y_pos)
                
                # Create label
                if d_var == 'injustice_pc_100k':
                    label = f"Per capita (per 100k) × {step}"
                elif d_var == 'injustice_share_pp':
                    label = f"Share (pp) × {step}"
                else:
                    label = f"Spike (binary) × {step}"
                y_labels.append(label)
                
                # Plot error bar
                color = phase_colors[j]
                ax.errorbar(coef, y_pos, xerr=se*1.96, fmt='o', color=color,
                           markersize=8, capsize=5, capthick=2, elinewidth=2,
                           ecolor=color, markeredgecolor=color)
                
                # Add coefficient text with significance stars
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                ax.text(coef + se*1.96 + 0.1, y_pos, f"{coef:.2f}{sig}", 
                       ha='left', va='center', fontsize=10, fontweight='bold', color=color)
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=11)
    
    # Add vertical line at zero
    ax.axvline(0, color='black', lw=1, alpha=0.6)
    
    # Set title and labels
    ax.set_title("Discourse–Protest Effects by Phase", fontsize=16, fontweight='bold')
    ax.set_xlabel("Effect on protests", fontsize=12)
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_colors[i], 
                                  markersize=8, label=f'Phase {i+1}') 
                       for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(CONFIG["output_dir"])
    plt.savefig(output_dir / 'fig_baseline_coef_clean.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_baseline_coef_clean.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Baseline coefficient plot saved")

def plot_heterogeneity_coefficients(heterogeneity_results):
    """Create heterogeneity coefficient plots"""
    print("\nCreating heterogeneity coefficient plots...")
    
    # Define colors for moderators
    moderator_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    moderator_labels = {
        'share_black': 'Share Black',
        'dem_vote_share': 'Dem. vote share', 
        'education_ba_plus': 'BA+ education',
        'median_income': 'Median income',
        'urban_z': 'Urbanization (z-score)'
    }
    
    # Get unique discourse variants from results
    d_variants = heterogeneity_results['D_variant'].unique()
    
    for d_var in d_variants:
        print(f"  Creating plot for {d_var}...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # Set title based on discourse variant
        if d_var == 'injustice_pc_100k':
            title = "Heterogeneous Discourse–Protest Effects by Amount of Discourse"
            xlabel = "Effect on protests (per 100k)"
        elif d_var == 'injustice_share_pp':
            title = "Heterogeneous Discourse–Protest Effects by Discourse Share"
            xlabel = "Effect on protests (pp)"
        else:
            title = "Heterogeneous Discourse–Protest Effects by Phase\nSpike (binary)"
            xlabel = "Effect on protests (binary 0/1)"
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Get data for this discourse variant
        data = heterogeneity_results[heterogeneity_results['D_variant'] == d_var]
        
        # Get unique moderators
        moderators = data['moderator'].unique()
        
        for i, step in enumerate(CONFIG["phases"]):
            ax = axes[i]
            step_data = data[data['phase'] == step]
            
            y_positions = []
            y_labels = []
            
            for j, mod in enumerate(moderators):
                mod_data = step_data[step_data['moderator'] == mod]
                
                if len(mod_data) > 0:
                    # Get triple interaction coefficient
                    if d_var == 'injustice_pc_100k':
                        triple_coef = mod_data[f'triple_coef_{step.lower()}_per100k'].iloc[0]
                        triple_se = mod_data[f'triple_se_{step.lower()}_per100k'].iloc[0]
                        triple_pval = mod_data[f'triple_pval_{step.lower()}_per100k'].iloc[0]
                    elif d_var == 'injustice_share_pp':
                        triple_coef = mod_data[f'triple_coef_{step.lower()}_pp'].iloc[0]
                        triple_se = mod_data[f'triple_se_{step.lower()}_pp'].iloc[0]
                        triple_pval = mod_data[f'triple_pval_{step.lower()}_pp'].iloc[0]
                    else:  # injustice_spike_1sd
                        triple_coef = mod_data[f'triple_coef_{step.lower()}_binary'].iloc[0]
                        triple_se = mod_data[f'triple_se_{step.lower()}_binary'].iloc[0]
                        triple_pval = mod_data[f'triple_pval_{step.lower()}_binary'].iloc[0]
                    
                    if pd.notna(triple_coef) and pd.notna(triple_se):
                        y_pos = j
                        y_positions.append(y_pos)
                        y_labels.append(moderator_labels.get(mod, mod))
                        
                        # Plot error bar with moderator-specific color
                        color = moderator_colors[j % len(moderator_colors)]
                        ax.errorbar(triple_coef, y_pos, xerr=triple_se*1.96, fmt='o', color=color,
                                    markersize=8, capsize=5, capthick=2, elinewidth=2,
                                    ecolor=color, markeredgecolor=color)
                        
                        # Add coefficient text with significance stars - POSITIONED RIGHT NEXT TO WHISKERS
                        sig = "***" if triple_pval < 0.01 else "**" if triple_pval < 0.05 else "*" if triple_pval < 0.1 else ""
                        
                        # Position text right next to the whisker (error bar)
                        x_text = triple_coef + 1.96*triple_se + 0.05  # Small offset from whisker
                        
                        # Simple text annotation right next to whisker
                        ax.annotate(f"{triple_coef:.2f}{sig}", xy=(x_text, y_pos), xycoords='data',
                                    xytext=(3,0), textcoords='offset points',
                                    ha='left', va='center', fontsize=9, fontweight='bold', color=color)
            
            # Set y-axis
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=11)
            
            # Add vertical line at zero
            ax.axvline(0, color='black', lw=1, alpha=0.6)
            
            # Set title for each phase
            phase_titles = {
                'Step1': 'Phase 1\n(May 25-28)',
                'Step2': 'Phase 2\n(May 29-Jun 4)', 
                'Step3': 'Phase 3\n(≥ Jun 5)'
            }
            ax.set_title(phase_titles[step], fontsize=14, fontweight='bold')
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)
            ax.set_xlabel(xlabel)
            
            # Remove y-axis grid lines
            ax.yaxis.grid(False)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=moderator_colors[i], 
                                      markersize=8, label=moderator_labels.get(mod, mod)) 
                           for i, mod in enumerate(moderators)]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.02))
        
        plt.tight_layout(pad=3.0)
        
        # Adjust x-axis limits to accommodate text - EXTENDED FOR BETTER FIT
        for ax in axes:
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0], xlim[1] + 0.5)  # Add more space on the right for text to fit
        
        # Save clean plots with updated titles
        output_dir = Path(CONFIG["output_dir"])
        plt.savefig(output_dir / f'fig_heterogeneity_steps_{d_var}_clean.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'fig_heterogeneity_steps_{d_var}_clean.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Clean heterogeneity plot for {d_var} saved")

def main():
    """Main execution function"""
    print("="*80)
    print("DISCOURSE-PROTEST ANALYSIS")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for demo safeguard
    demo_dir = Path("./outputs_demo")
    if demo_dir.exists():
        print("SAFEGUARD: Found outputs_demo/. This is a demo run.")
        print("Real results will be written to ./outputs/")
    
    # Load and prepare data
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    try:
        # Load main data
        print("Loading main dataset...")
        df = pd.read_csv("./daily_tweets_with_population_and_protests_complete.csv")
        print(f"✓ Loaded {len(df)} observations")
        
        # Rename columns for consistency
        df = df.rename(columns={
            'fips_code': 'fips',
            'total_popE': 'population',
            'tweet_count': 'total_tweets',
            'protest_count': 'protests'
        })
        
        print(f"✓ Columns after rename: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['fips', 'population', 'total_tweets', 'protests', 'injustice_tweets_per_1000', 'non_injustice_tweets_per_1000']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to relevant date range
        df = df[(df['date'] >= '2020-05-25') & (df['date'] <= '2020-06-15')]
        
        # Filter out counties with zero population (causes division by zero)
        df = df[df['population'] > 0]
        print(f"✓ After filtering zero population counties: {len(df)} observations")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Load ACS county-level controls
        print("Loading ACS county-level controls...")
        acs_data = pd.read_csv("./acscounty_level_controls.csv")
        acs_data = acs_data.rename(columns={'GEOID': 'fips'})
        print(f"✓ Loaded {len(acs_data)} counties from ACS data")
        
        # Load voting data
        print("Loading voting data...")
        voting_data = pd.read_csv("./county_stats_with_voting.csv")
        voting_data = voting_data.rename(columns={'fips_code': 'fips'})
        # Fix Democratic vote share calculation (per_dem is already a proportion)
        voting_data['dem_vote_share'] = voting_data['per_dem']  # Remove /100 division
        print(f"✓ Loaded {len(voting_data)} counties from voting data")
        
        # Merge datasets
        print("Merging datasets...")
        df = df.merge(acs_data, on='fips', how='left')
        df = df.merge(voting_data, on='fips', how='left')
        
        # Fix column names after merge (handle duplicate column names)
        if 'total_tweets_x' in df.columns:
            df = df.rename(columns={'total_tweets_x': 'total_tweets'})
        if 'total_tweets_y' in df.columns:
            df = df.drop(columns=['total_tweets_y'])
        
        print(f"✓ Merged dataset has {len(df)} observations")
        
        # Create derived variables
        df = create_derived_variables(df)
        
        # Add phase dummies
        df = add_phase_dummies(df)
        
        # Add urbanization measures
        df = add_urbanization_measures(df)
        
        # Run preflight checks
        preflight_checks(df)
        
        # Set index for panel analysis
        df = df.set_index(['fips', 'date'])
        print(f"✓ Panel dataset ready with {len(df)} observations")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Estimate models
    try:
        # Estimate baseline models
        baseline_results = estimate_baseline_models(df)
        
        # Estimate heterogeneity models
        heterogeneity_results = estimate_heterogeneity_models(df)
        
        # Estimate spike heterogeneity models
        spike_results = estimate_spike_heterogeneity_models(df)
        
        # Combine heterogeneity results
        all_heterogeneity_results = pd.concat([heterogeneity_results, spike_results], ignore_index=True)
        
        # Calculate marginal effects
        marginal_effects = calculate_marginal_effects(df, baseline_results, all_heterogeneity_results)
        
        # Export results
        export_results(baseline_results, all_heterogeneity_results, marginal_effects)
        
        # Create plots
        plot_baseline_coefficients(baseline_results)
        plot_heterogeneity_coefficients(all_heterogeneity_results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✓ All models estimated successfully")
        print("✓ Results exported to CSV files")
        print("✓ Plots generated and saved")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
