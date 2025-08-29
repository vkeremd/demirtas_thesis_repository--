#!/usr/bin/env python3
"""
Appendix I: Complete Code for BERTweet Model Training and Analysis
================================================================

This file documents the complete workflow from raw tweet data processing through
BERTweet model training to the final analysis dataset. The code is organized
chronologically and includes detailed comments explaining each step.

WORKFLOW OVERVIEW:
1. Raw tweet data processing and location extraction
2. Training data preparation for BERTweet model
3. BERTweet model training and validation
4. Tweet classification using trained model
5. Data aggregation and combination with protest data
6. Final dataset creation

Author: [Your Name]
Date: [Current Date]
Thesis: Discourse and Protest Activity Analysis
"""

# =============================================================================
# SECURITY AND CREDENTIALS
# =============================================================================

# IMPORTANT: This code does NOT contain any API keys, passwords, or sensitive credentials
# All data processing is done on local files and publicly available datasets
# If you need to use Twitter API or other services, add your credentials securely:
# - Use environment variables: os.getenv('TWITTER_API_KEY')
# - Use .env files (not included in version control)
# - Never hardcode credentials in the code

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
from textblob import TextBlob
from collections import defaultdict
import re
import csv
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 COMPLETE BERTWEET WORKFLOW DOCUMENTATION")
print("=" * 80)

# =============================================================================
# STEP 1: RAW TWEET DATA PROCESSING AND LOCATION EXTRACTION
# =============================================================================

def process_raw_tweet_data():
    """
    Step 1: Process raw tweet data and extract location information
    
    This function handles the initial processing of raw tweet data from JSON files,
    extracts location information, and prepares the data for further analysis.
    The location data is crucial for linking tweets to specific counties for
    the protest analysis.
    """
    
    print("\n📊 STEP 1: RAW TWEET DATA PROCESSING")
    print("-" * 50)
    
    # Load location-to-county mapping data
    print("Loading location-to-county mapping data...")
    filtered_entries = []
    with open("loc-to-county-v2022.01.23.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            # Only keep entries with valid county codes
            if "county" in entry and entry["county"].isdigit():
                filtered_entries.append(entry)
    
    # Convert to DataFrame for easier processing
    location_df = pd.DataFrame(filtered_entries)
    print(f"Loaded {len(location_df)} valid location entries")
    
    # Create a dictionary for fast lookup
    places_dict = {}
    for _, row in location_df.iterrows():
        if 'place_id' in row and pd.notna(row['place_id']):
            places_dict[row['place_id']] = {
                'county': row.get('county'),
                'state': row.get('state'),
                'place_name': row.get('place_name'),
                'place_full_name': row.get('place_full_name')
            }
    
    print(f"Created lookup dictionary with {len(places_dict)} places")
    
    def extract_location_info(tweet_data, places_dict):
        """Extract location information from a tweet using the places dictionary"""
        location_info = {
            'place_id': None,
            'place_name': None,
            'place_full_name': None,
            'place_country': None,
            'place_country_code': None,
            'place_type': None,
            'county': None,
            'state': None
        }
        
        # Extract place information from tweet
        if 'place' in tweet_data and tweet_data['place']:
            place = tweet_data['place']
            location_info['place_id'] = place.get('id')
            location_info['place_name'] = place.get('name')
            location_info['place_full_name'] = place.get('full_name')
            location_info['place_country'] = place.get('country')
            location_info['place_country_code'] = place.get('country_code')
            location_info['place_type'] = place.get('place_type')
            
            # Look up county and state information
            if location_info['place_id'] in places_dict:
                county_info = places_dict[location_info['place_id']]
                location_info['county'] = county_info.get('county')
                location_info['state'] = county_info.get('state')
        
        return location_info
    
    # Process raw tweet files
    print("Processing raw tweet data...")
    processed_tweets = []
    
    # This would typically process multiple tweet files
    # For demonstration, we show the structure
    tweet_files = ["tweets_2020_05.json", "tweets_2020_06.json"]  # Example files
    
    for file_path in tweet_files:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        tweet = json.loads(line)
                        if 'data' in tweet and tweet['data']:
                            tweet_data = tweet['data']
                            
                            # Extract basic tweet information
                            tweet_info = {
                                'tweet_id': tweet_data.get('id'),
                                'text': tweet_data.get('text'),
                                'created_at': tweet_data.get('created_at'),
                                'author_id': tweet_data.get('author_id')
                            }
                            
                            # Extract location information
                            location_info = extract_location_info(tweet_data, places_dict)
                            tweet_info.update(location_info)
                            
                            processed_tweets.append(tweet_info)
                            
                    except json.JSONDecodeError:
                        continue
    
    print(f"Processed {len(processed_tweets)} tweets with location data")
    
    # Save processed data
    processed_df = pd.DataFrame(processed_tweets)
    processed_df.to_csv("processed_tweets_with_location.csv", index=False)
    print("Saved processed tweets to 'processed_tweets_with_location.csv'")
    
    return processed_df

# =============================================================================
# STEP 2: TRAINING DATA PREPARATION FOR BERTWEET MODEL
# =============================================================================

def prepare_training_data():
    """
    Step 2: Prepare training data for BERTweet model
    
    This function creates the training dataset for the BERTweet model by
    manually labeling a subset of tweets as either containing injustice discourse
    (label=1) or not (label=0). This labeled data is essential for supervised
    learning of the injustice classification model.
    """
    
    print("\n📝 STEP 2: TRAINING DATA PREPARATION")
    print("-" * 50)
    
    # Load processed tweet data
    print("Loading processed tweet data...")
    df = pd.read_csv("processed_tweets_with_location.csv")
    print(f"Loaded {len(df)} processed tweets")
    
    # Create training dataset with manual labels
    # This is a simplified version - in practice, you would manually label tweets
    print("Creating training dataset with manual labels...")
    
    # Example of how training data would be created
    # In practice, this would involve manual annotation of tweets
    training_examples = [
        # Injustice-related tweets (label=1)
        {"text": "Another unarmed Black man killed by police. This is systemic racism.", "label": 1},
        {"text": "Justice for George Floyd. The system is broken.", "label": 1},
        {"text": "Police brutality must end. Black lives matter.", "label": 1},
        {"text": "Institutional racism is real and deadly.", "label": 1},
        {"text": "The criminal justice system is biased against people of color.", "label": 1},
        
        # Non-injustice tweets (label=0)
        {"text": "Great weather today! Perfect for a walk.", "label": 0},
        {"text": "Just finished reading a good book.", "label": 0},
        {"text": "Happy birthday to my friend!", "label": 0},
        {"text": "The new restaurant downtown is amazing.", "label": 0},
        {"text": "Can't wait for the weekend.", "label": 0}
    ]
    
    # Create training DataFrame
    training_df = pd.DataFrame(training_examples)
    
    # In practice, you would load your actual training data
    # training_df = pd.read_csv("manual_training_data.csv")
    
    print(f"Created training dataset with {len(training_df)} examples")
    print(f"Label distribution: {training_df['label'].value_counts().to_dict()}")
    
    # Save training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_filename = f"training_data_{timestamp}.csv"
    training_df.to_csv(training_filename, index=False)
    print(f"Saved training data to '{training_filename}'")
    
    return training_df, training_filename

# =============================================================================
# STEP 3: BERTWEET MODEL TRAINING AND VALIDATION
# =============================================================================

def train_bertweet_model(training_filename):
    """
    Step 3: Train BERTweet model for injustice classification
    
    This function trains a BERTweet model to classify tweets as containing
    injustice discourse or not. BERTweet is a pre-trained language model
    specifically designed for Twitter text, making it ideal for this task.
    The model learns to recognize patterns in language that indicate discussions
    of systemic racism, police brutality, and other forms of injustice.
    """
    
    print("\n🤖 STEP 3: BERTWEET MODEL TRAINING")
    print("-" * 50)
    
    # Load training data
    print("Loading training data...")
    df = pd.read_csv(training_filename)
    print(f"Training dataset size: {len(df)} tweets")
    
    # Clean and prepare data
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)  # Ensure labels are integers
    
    print(f"After cleaning: {len(df)} tweets")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    
    print(f"Training set: {len(train_df)} tweets")
    print(f"Validation set: {len(val_df)} tweets")
    
    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    # Load BERTweet tokenizer and model
    print("Loading BERTweet tokenizer and model...")
    model_name = "vinai/bertweet-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./bertweet_injustice_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model performance...")
    results = trainer.evaluate()
    print(f"Validation accuracy: {results['eval_accuracy']:.4f}")
    print(f"Validation loss: {results['eval_loss']:.4f}")
    
    # Save the trained model
    model_save_path = "./bertweet_injustice_final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to '{model_save_path}'")
    
    return model_save_path

# =============================================================================
# STEP 4: TWEET CLASSIFICATION USING TRAINED MODEL
# =============================================================================

def classify_all_tweets(model_path):
    """
    Step 4: Classify all tweets using the trained BERTweet model
    
    This function applies the trained BERTweet model to classify all tweets
    in the dataset as either containing injustice discourse (1) or not (0).
    The classification is done in batches for efficiency, and the results
    are saved for further analysis.
    """
    
    print("\n🔍 STEP 4: TWEET CLASSIFICATION")
    print("-" * 50)
    
    # Load trained model and tokenizer
    print("Loading trained BERTweet model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    def classify_tweets_batch(tweets, batch_size=32):
        """Classify tweets in batches for efficiency"""
        predictions = []
        
        for i in range(0, len(tweets), batch_size):
            batch_tweets = tweets[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_tweets, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(batch_predictions)
        
        return predictions
    
    # Load all tweets for classification
    print("Loading all tweets for classification...")
    all_tweets_df = pd.read_csv("processed_tweets_with_location.csv")
    print(f"Loaded {len(all_tweets_df)} tweets for classification")
    
    # Classify tweets
    print("Classifying tweets...")
    tweet_texts = all_tweets_df['text'].tolist()
    predictions = classify_tweets_batch(tweet_texts)
    
    # Add predictions to dataframe
    all_tweets_df['injustice_label'] = predictions
    all_tweets_df['is_injustice'] = all_tweets_df['injustice_label'] == 1
    
    print(f"Classification complete!")
    print(f"Injustice tweets: {all_tweets_df['is_injustice'].sum()}")
    print(f"Non-injustice tweets: {len(all_tweets_df) - all_tweets_df['is_injustice'].sum()}")
    
    # Save classified tweets
    classified_filename = "classified_tweets_with_location.csv"
    all_tweets_df.to_csv(classified_filename, index=False)
    print(f"Saved classified tweets to '{classified_filename}'")
    
    return all_tweets_df

# =============================================================================
# STEP 5: DATA AGGREGATION AND DAILY COUNTS
# =============================================================================

def aggregate_daily_tweet_counts(classified_df):
    """
    Step 5: Aggregate tweet counts by county and date
    
    This function aggregates the classified tweets by county and date to create
    daily counts of injustice and non-injustice tweets. This aggregation is
    necessary for the time-series analysis of the relationship between discourse
    and protest activity.
    """
    
    print("\n📊 STEP 5: DAILY TWEET COUNT AGGREGATION")
    print("-" * 50)
    
    # Filter tweets with valid county information
    print("Filtering tweets with valid county information...")
    valid_counties_df = classified_df[
        (classified_df['county'].notna()) & 
        (classified_df['county'] != '') &
        (classified_df['state'].notna()) & 
        (classified_df['state'] != '')
    ].copy()
    
    print(f"Tweets with valid county info: {len(valid_counties_df)}")
    
    # Convert date column
    valid_counties_df['created_at'] = pd.to_datetime(valid_counties_df['created_at'])
    valid_counties_df['date'] = valid_counties_df['created_at'].dt.date
    
    # Create FIPS code (county identifier)
    valid_counties_df['fips_code'] = (
        valid_counties_df['state'].astype(str).str.zfill(2) + 
        valid_counties_df['county'].astype(str).str.zfill(3)
    )
    
    # Aggregate by county, date, and injustice label
    print("Aggregating tweet counts by county, date, and injustice label...")
    daily_counts = valid_counties_df.groupby([
        'fips_code', 'county', 'state', 'date', 'injustice_label'
    ]).size().reset_index(name='tweet_count')
    
    # Pivot to get separate columns for injustice and non-injustice tweets
    daily_counts_pivot = daily_counts.pivot_table(
        index=['fips_code', 'county', 'state', 'date'],
        columns='injustice_label',
        values='tweet_count',
        fill_value=0
    ).reset_index()
    
    # Rename columns
    daily_counts_pivot.columns.name = None
    daily_counts_pivot = daily_counts_pivot.rename(columns={
        0: 'non_injustice_tweets',
        1: 'injustice_tweets'
    })
    
    # Add total tweet count
    daily_counts_pivot['total_tweets'] = (
        daily_counts_pivot['injustice_tweets'] + 
        daily_counts_pivot['non_injustice_tweets']
    )
    
    print(f"Created daily counts for {len(daily_counts_pivot)} county-date combinations")
    
    # Save daily counts
    daily_counts_filename = "daily_tweet_counts_by_county.csv"
    daily_counts_pivot.to_csv(daily_counts_filename, index=False)
    print(f"Saved daily counts to '{daily_counts_filename}'")
    
    return daily_counts_pivot

# =============================================================================
# STEP 6: ADD POPULATION DATA
# =============================================================================

def add_population_data(daily_counts_df):
    """
    Step 6: Add population data to normalize tweet counts
    
    This function adds population data for each county to enable per-capita
    analysis of tweet activity. This normalization is important because
    larger counties will naturally have more tweets, and we want to compare
    discourse intensity across counties of different sizes.
    """
    
    print("\n👥 STEP 6: ADDING POPULATION DATA")
    print("-" * 50)
    
    # Load population data (example - you would use actual Census data)
    print("Loading population data...")
    
    # This is a simplified example - in practice, you would load actual Census data
    # population_df = pd.read_csv("county_population_2020.csv")
    
    # For demonstration, create sample population data
    sample_population_data = {
        'fips_code': ['10001', '10003', '10005', '11001', '12001'],
        'county_name': ['Kent', 'New Castle', 'Sussex', 'District of Columbia', 'Alachua'],
        'state_name': ['DE', 'DE', 'DE', 'DC', 'FL'],
        'total_popE': [165030, 558753, 234225, 689545, 269043]
    }
    population_df = pd.DataFrame(sample_population_data)
    
    print(f"Loaded population data for {len(population_df)} counties")
    
    # Merge population data with daily counts
    print("Merging population data with daily counts...")
    daily_with_pop = daily_counts_df.merge(
        population_df[['fips_code', 'county_name', 'state_name', 'total_popE']],
        on='fips_code',
        how='left'
    )
    
    # Calculate per-capita rates (per 1000 population)
    daily_with_pop['tweets_per_1000'] = (
        daily_with_pop['total_tweets'] / daily_with_pop['total_popE'] * 1000
    )
    daily_with_pop['injustice_tweets_per_1000'] = (
        daily_with_pop['injustice_tweets'] / daily_with_pop['total_popE'] * 1000
    )
    daily_with_pop['non_injustice_tweets_per_1000'] = (
        daily_with_pop['non_injustice_tweets'] / daily_with_pop['total_popE'] * 1000
    )
    
    print("Calculated per-capita tweet rates")
    
    # Save data with population
    population_filename = "daily_tweets_with_population_final_ultimate.csv"
    daily_with_pop.to_csv(population_filename, index=False)
    print(f"Saved data with population to '{population_filename}'")
    
    return daily_with_pop

# =============================================================================
# STEP 7: COMBINE WITH PROTEST DATA
# =============================================================================

def combine_with_protest_data(daily_with_pop_df):
    """
    Step 7: Combine tweet data with protest event data
    
    This function merges the daily tweet data with protest event data from
    ACLED (Armed Conflict Location & Event Data Project). This creates the
    final dataset that links discourse intensity with protest activity,
    enabling the analysis of the relationship between social media discourse
    and collective action.
    """
    
    print("\n🏛️ STEP 7: COMBINING WITH PROTEST DATA")
    print("-" * 50)
    
    # Load protest data
    print("Loading protest data from ACLED...")
    protests_df = pd.read_csv('protests_acled_Data.csv', quoting=1)  # QUOTE_ALL
    print(f"Loaded {len(protests_df)} protest events")
    
    # Check protest data structure
    print(f"Protest data columns: {list(protests_df.columns)}")
    print(f"Date range: {protests_df['event_date'].min()} to {protests_df['event_date'].max()}")
    
    # Clean and prepare protest data
    protests_df['event_date'] = pd.to_datetime(protests_df['event_date'])
    protests_df['date'] = protests_df['event_date'].dt.date
    
    # Create FIPS codes for protest data
    protests_df['fips_code'] = (
        protests_df['admin1'].astype(str).str.zfill(2) + 
        protests_df['admin2'].astype(str).str.zfill(3)
    )
    
    # Aggregate protest counts by county and date
    print("Aggregating protest counts by county and date...")
    daily_protests = protests_df.groupby(['fips_code', 'date']).size().reset_index(name='protest_count')
    
    print(f"Created daily protest counts for {len(daily_protests)} county-date combinations")
    
    # Merge tweet data with protest data
    print("Merging tweet data with protest data...")
    daily_tweets_df = daily_with_pop_df.copy()
    daily_tweets_df['date'] = pd.to_datetime(daily_tweets_df['date']).dt.date
    
    # Perform the merge
    combined_df = daily_tweets_df.merge(
        daily_protests[['fips_code', 'date', 'protest_count']],
        on=['fips_code', 'date'],
        how='left'
    )
    
    # Fill missing protest counts with 0
    combined_df['protest_count'] = combined_df['protest_count'].fillna(0)
    
    print(f"Combined dataset has {len(combined_df)} observations")
    print(f"Counties with protests: {len(combined_df[combined_df['protest_count'] > 0])}")
    
    # Add date components for analysis
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['year'] = combined_df['date'].dt.year
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['day'] = combined_df['date'].dt.day
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    
    # Save final combined dataset
    final_filename = "daily_tweets_with_population_and_protests_complete.csv"
    combined_df.to_csv(final_filename, index=False)
    print(f"Saved final combined dataset to '{final_filename}'")
    
    # Print summary statistics
    print("\n📈 FINAL DATASET SUMMARY:")
    print(f"Total observations: {len(combined_df):,}")
    print(f"Unique counties: {combined_df['fips_code'].nunique()}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Total injustice tweets: {combined_df['injustice_tweets'].sum():,}")
    print(f"Total protests: {combined_df['protest_count'].sum():,}")
    print(f"Counties with protests: {(combined_df['protest_count'] > 0).sum()}")
    
    return combined_df

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that executes the complete workflow from raw data to final output
    
    This function orchestrates the entire process, calling each step in sequence
    and providing progress updates throughout the pipeline.
    """
    
    print("🚀 STARTING COMPLETE BERTWEET WORKFLOW")
    print("=" * 80)
    print("This workflow processes raw tweet data, trains a BERTweet model for")
    print("injustice classification, and creates the final analysis dataset.")
    print("=" * 80)
    
    try:
        # Step 1: Process raw tweet data
        processed_df = process_raw_tweet_data()
        
        # Step 2: Prepare training data
        training_df, training_filename = prepare_training_data()
        
        # Step 3: Train BERTweet model
        model_path = train_bertweet_model(training_filename)
        
        # Step 4: Classify all tweets
        classified_df = classify_all_tweets(model_path)
        
        # Step 5: Aggregate daily counts
        daily_counts_df = aggregate_daily_tweet_counts(classified_df)
        
        # Step 6: Add population data
        daily_with_pop_df = add_population_data(daily_counts_df)
        
        # Step 7: Combine with protest data
        final_df = combine_with_protest_data(daily_with_pop_df)
        
        print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Final output file: daily_tweets_with_population_and_protests_complete.csv")
        print("This file contains:")
        print("- Daily tweet counts by county (injustice and non-injustice)")
        print("- Population data for per-capita analysis")
        print("- Protest event counts by county and date")
        print("- Date components for time-series analysis")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR IN WORKFLOW: {str(e)}")
        print("Please check the error and try again.")
        raise

# =============================================================================
# DOCUMENTATION AND EXPLANATION
# =============================================================================

def explain_workflow():
    """
    Provide detailed explanation of the workflow and its components
    
    This function explains the rationale behind each step and how the
    components work together to create the final analysis dataset.
    """
    
    print("\n📚 WORKFLOW EXPLANATION")
    print("=" * 80)
    
    print("\n🎯 OVERALL OBJECTIVE:")
    print("The goal is to analyze the relationship between social media discourse")
    print("about injustice and protest activity during the Black Lives Matter movement.")
    print("This requires processing millions of tweets, training a machine learning")
    print("model to identify injustice-related content, and linking this to protest data.")
    
    print("\n🔧 KEY COMPONENTS:")
    
    print("\n1. BERTweet Model:")
    print("   - Pre-trained language model specifically designed for Twitter text")
    print("   - Fine-tuned to classify tweets as injustice-related or not")
    print("   - Handles Twitter-specific language patterns, emojis, and abbreviations")
    
    print("\n2. Location Processing:")
    print("   - Links tweets to specific counties using FIPS codes")
    print("   - Enables geographic analysis of discourse patterns")
    print("   - Critical for matching with protest event locations")
    
    print("\n3. Time-Series Aggregation:")
    print("   - Converts individual tweets to daily county-level counts")
    print("   - Enables analysis of discourse intensity over time")
    print("   - Matches the temporal structure of protest data")
    
    print("\n4. Population Normalization:")
    print("   - Converts raw counts to per-capita rates")
    print("   - Enables fair comparison across counties of different sizes")
    print("   - Provides interpretable measures of discourse intensity")
    
    print("\n5. Protest Data Integration:")
    print("   - Combines discourse data with ACLED protest event data")
    print("   - Creates the final dataset for econometric analysis")
    print("   - Enables testing of discourse-protest relationships")
    
    print("\n📊 FINAL DATASET STRUCTURE:")
    print("The final dataset contains one row per county-date combination with:")
    print("- fips_code: County identifier")
    print("- date: Date of observation")
    print("- injustice_tweets: Count of injustice-related tweets")
    print("- total_tweets: Total tweet count")
    print("- tweets_per_1000: Per-capita tweet rate")
    print("- injustice_tweets_per_1000: Per-capita injustice tweet rate")
    print("- protest_count: Number of protests on that date")
    print("- Population and geographic variables")
    
    print("\n🎓 ACADEMIC SIGNIFICANCE:")
    print("This workflow enables rigorous empirical analysis of how social media")
    print("discourse about injustice relates to collective action. The BERTweet")
    print("model provides a sophisticated measure of discourse intensity that")
    print("captures the nuanced language patterns of social media conversations.")
    print("The geographic and temporal structure allows for causal inference")
    print("about the relationship between online discourse and offline protest.")
    
    print("\n🔒 SECURITY AND DATA SOURCES:")
    print("This workflow uses only publicly available datasets and local processing.")
    print("No API keys or sensitive credentials are included in the code.")
    print("Data sources include:")
    print("- Public tweet datasets (JSON files)")
    print("- Census population data (public)")
    print("- ACLED protest data (public)")
    print("- Location mapping data (public)")
    print("For production use, implement secure credential management.")

if __name__ == "__main__":
    # Run the complete workflow
    main()
    
    # Provide detailed explanation
    explain_workflow()
