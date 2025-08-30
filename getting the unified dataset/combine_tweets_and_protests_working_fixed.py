#!/usr/bin/env python3
"""
Working combination of tweet data with protest data.
Handles all FIPS code format differences and creates the final analysis dataset.
"""

import pandas as pd
import numpy as np

def combine_tweets_and_protests_working():
    """Combine tweet data with protest data with proper FIPS code handling."""
    
    print("=" * 80)
    print("WORKING COMBINATION OF TWEETS AND PROTESTS DATA")
    print("=" * 80)
    
    # Load the tweet data
    print("\n1. Loading tweet data...")
    tweets_df = pd.read_csv('daily_tweets_with_population_final_ultimate.csv')
    print(f"   Tweet records: {len(tweets_df):,}")
    
    # Load the protest data with proper CSV parsing
    print("\n2. Loading protest data with robust parsing...")
    protests_df = pd.read_csv('protests_acled_Data.csv', quoting=1)  # QUOTE_ALL
    print(f"   Protest records: {len(protests_df):,}")
    
    # Check the protest data structure
    print(f"\n3. Examining protest data structure...")
    print(f"   Protest data columns: {list(protests_df.columns)}")
    print(f"   Date range: {protests_df['event_date'].min()} to {protests_df['event_date'].max()}")
    
    # Clean and prepare protest data
    print("\n4. Cleaning and preparing protest data...")
    protests_df['event_date'] = pd.to_datetime(protests_df['event_date'])
    protests_df['date'] = protests_df['event_date'].dt.date
    
    # Create FIPS codes for protest data
    print("\n5. Creating FIPS codes for protest data...")
    protests_df['fips_code'] = (
        protests_df['admin1'].astype(str).str.zfill(2) + 
        protests_df['admin2'].astype(str).str.zfill(3)
    )
    
    # Aggregate protest counts by county and date
    print("\n6. Aggregating protest counts by county and date...")
    daily_protests = protests_df.groupby(['fips_code', 'date']).size().reset_index(name='protest_count')
    print(f"   Daily protest combinations: {len(daily_protests):,}")
    
    # Prepare tweet data for merging
    print("\n7. Preparing tweet data for merging...")
    daily_tweets_df = tweets_df.copy()
    daily_tweets_df['date'] = pd.to_datetime(daily_tweets_df['date']).dt.date
    
    # Perform the merge
    print("\n8. Merging tweet data with protest data...")
    combined_df = daily_tweets_df.merge(
        daily_protests[['fips_code', 'date', 'protest_count']],
        on=['fips_code', 'date'],
        how='left'
    )
    
    # Fill missing protest counts with 0
    combined_df['protest_count'] = combined_df['protest_count'].fillna(0)
    
    print(f"\n9. Merge results:")
    print(f"   Combined dataset size: {len(combined_df):,}")
    print(f"   Counties with protests: {len(combined_df[combined_df['protest_count'] > 0])}")
    print(f"   Total protests: {combined_df['protest_count'].sum():,}")
    
    # Add date components for analysis
    print("\n10. Adding date components for analysis...")
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['year'] = combined_df['date'].dt.year
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['day'] = combined_df['date'].dt.day
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    
    # Save final combined dataset
    print("\n11. Saving final combined dataset...")
    final_filename = "daily_tweets_with_population_and_protests_complete.csv"
    combined_df.to_csv(final_filename, index=False)
    print(f"   Saved to: {final_filename}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Total observations: {len(combined_df):,}")
    print(f"Unique counties: {combined_df['fips_code'].nunique()}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Total injustice tweets: {combined_df['injustice_tweets'].sum():,}")
    print(f"Total protests: {combined_df['protest_count'].sum():,}")
    print(f"Counties with protests: {(combined_df['protest_count'] > 0).sum()}")
    
    # Show sample of the data
    print("\n" + "=" * 80)
    print("SAMPLE OF FINAL DATASET")
    print("=" * 80)
    print(combined_df.head(10).to_string())
    
    return combined_df

def verify_merge_quality(combined_df):
    """Verify the quality of the merged dataset."""
    
    print("\n" + "=" * 80)
    print("MERGE QUALITY VERIFICATION")
    print("=" * 80)
    
    # Check for missing data
    print("\n1. Checking for missing data...")
    missing_counts = combined_df.isnull().sum()
    print("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"   {col}: {count}")
    
    # Check data types
    print("\n2. Checking data types...")
    print(combined_df.dtypes)
    
    # Check for duplicates
    print("\n3. Checking for duplicates...")
    duplicates = combined_df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check protest data distribution
    print("\n4. Checking protest data distribution...")
    protest_stats = combined_df['protest_count'].describe()
    print("Protest count statistics:")
    print(protest_stats)
    
    # Check tweet data distribution
    print("\n5. Checking tweet data distribution...")
    tweet_stats = combined_df['injustice_tweets'].describe()
    print("Injustice tweet statistics:")
    print(tweet_stats)
    
    print("\n✅ Merge quality verification complete!")

if __name__ == "__main__":
    # Run the combination
    combined_df = combine_tweets_and_protests_working()
    
    # Verify the merge quality
    verify_merge_quality(combined_df)
    
    print("\n" + "=" * 80)
    print("COMBINATION COMPLETE")
    print("=" * 80)
    print("Tweet and protest data successfully combined!")
    print("Final dataset ready for analysis.")
