# CORD-19 Full Analysis Notebook

# Part 1: Data Loading and Basic Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import streamlit as st

# Download and load the data
print("Loading CORD-19 metadata...")
# Note: You'll need to download metadata.csv from https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
# and place it in the same directory as this notebook
df = pd.read_csv('metadata.csv', low_memory=False)

print("Data loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Examine the first few rows and data structure
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

# Basic data exploration
print(f"\nDataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nData types:")
print(df.dtypes)

print("\nMissing values in important columns:")
important_columns = ['title', 'abstract', 'journal', 'publish_time', 'authors', 'doi']
for col in important_columns:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        print(f"{col}: {missing_count} missing ({missing_percent:.2f}%)")

print("\nBasic statistics for numerical columns:")
print(df.describe())

# Part 2: Data Cleaning and Preparation
print("\n--- DATA CLEANING ---")

# Create a copy for cleaning
df_clean = df.copy()

# Handle missing data strategy
print("\nHandling missing data:")
# For title - we'll keep only rows with titles since they're essential
initial_count = len(df_clean)
df_clean = df_clean.dropna(subset=['title'])
print(f"Removed {initial_count - len(df_clean)} rows with missing titles")

# For abstract - we'll keep them but note they're missing
abstract_missing = df_clean['abstract'].isnull().sum()
print(f"Abstracts missing in {abstract_missing} rows ({abstract_missing/len(df_clean)*100:.2f}%)")

# For journal - we'll fill with 'Unknown'
df_clean['journal'] = df_clean['journal'].fillna('Unknown')

# For publish_time - we'll extract what we can and drop problematic ones
print(f"Initial publish_time unique values sample: {df_clean['publish_time'].dropna().unique()[:10]}")

# Convert publish_time to datetime, handling various formats
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')

# Extract year from publication date
df_clean['year'] = df_clean['publish_time'].dt.year

# Handle rows where year couldn't be extracted
year_missing = df_clean['year'].isnull().sum()
print(f"Years missing in {year_missing} rows")

# For analysis purposes, we'll work with data that has years
df_clean = df_clean.dropna(subset=['year'])
df_clean['year'] = df_clean['year'].astype(int)

# Create new columns for analysis
df_clean['abstract_word_count'] = df_clean['abstract'].apply(
    lambda x: len(str(x).split()) if pd.notnull(x) else 0
)
df_clean['title_word_count'] = df_clean['title'].apply(
    lambda x: len(str(x).split()) if pd.notnull(x) else 0
)

print(f"\nCleaned dataset shape: {df_clean.shape}")
print(f"Final year range: {df_clean['year'].min()} to {df_clean['year'].max()}")

# Part 3: Data Analysis and Visualization
print("\n--- DATA ANALYSIS AND VISUALIZATION ---")

# Set up plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Count papers by publication year
year_counts = df_clean['year'].value_counts().sort_index()
axes[0, 0].bar(year_counts.index, year_counts.values, color='skyblue')
axes[0, 0].set_title('Number of Publications by Year', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Publications')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Identify top journals publishing COVID-19 research
top_journals = df_clean['journal'].value_counts().head(10)
axes[0, 1].barh(range(len(top_journals)), top_journals.values, color='lightgreen')
axes[0, 1].set_yticks(range(len(top_journals)))
axes[0, 1].set_yticklabels(top_journals.index)
axes[0, 1].set_title('Top 10 Journals by Publication Count', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Publications')

# 3. Most frequent words in titles
def clean_text(text):
    """Clean text by removing punctuation and common stop words"""
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Extract and clean titles
all_titles = ' '.join(df_clean['title'].apply(clean_text))
title_words = all_titles.split()
word_freq = Counter(title_words).most_common(20)

words, counts = zip(*word_freq)
axes[1, 0].barh(range(len(words)), counts, color='salmon')
axes[1, 0].set_yticks(range(len(words)))
axes[1, 0].set_yticklabels(words)
axes[1, 0].set_title('Top 20 Words in Paper Titles', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Frequency')

# 4. Distribution of abstract word counts
axes[1, 1].hist(df_clean['abstract_word_count'], bins=50, color='purple', alpha=0.7)
axes[1, 1].set_title('Distribution of Abstract Word Counts', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_xlim(0, 1000)  # Limit x-axis for better visualization

plt.tight_layout()
plt.show()

# Generate word cloud
print("\nGenerating word cloud...")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold')
plt.show()

# Additional analysis
print("\nAdditional insights:")
print(f"Total papers analyzed: {len(df_clean)}")
print(f"Time period covered: {df_clean['year'].min()} - {df_clean['year'].max()}")
print(f"Journal with most publications: {top_journals.index[0]} ({top_journals.iloc[0]} papers)")
print(f"Average abstract length: {df_clean['abstract_word_count'].mean():.1f} words")

# Part 4: Streamlit Application
print("\n--- STREAMLIT APPLICATION ---")

# Create a separate Python file for Streamlit app
streamlit_code = '''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# Set page configuration
st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide")

# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv', low_memory=False)
    # Apply the same cleaning as in the notebook
    df_clean = df.dropna(subset=['title'])
    df_clean['journal'] = df_clean['journal'].fillna('Unknown')
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['year'] = df_clean['publish_time'].dt.year
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    return df_clean

df = load_data()

# App title and description
st.title("CORD-19 Research Papers Explorer")
st.write("""
This interactive dashboard explores the COVID-19 Open Research Dataset (CORD-19), 
containing scientific papers related to COVID-19 and coronavirus research.
""")

# Sidebar for filters
st.sidebar.header("Filters")

# Year range slider
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2020, 2021)
)

# Journal selection
journals = ['All'] + df['journal'].value_counts().head(20).index.tolist()
selected_journal = st.sidebar.selectbox("Select Journal", journals)

# Apply filters
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_journal != 'All':
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    st.metric("Total Papers", len(filtered_df))
    st.metric("Years Covered", f"{year_range[0]} - {year_range[1]}")
    
    # Show sample data
    st.subheader("Sample Papers")
    st.dataframe(filtered_df[['title', 'journal', 'year']].head(10))

with col2:
    st.subheader("Publications by Year")
    year_counts = filtered_df['year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(year_counts.index, year_counts.values, color='skyblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Additional visualizations
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top Journals")
    top_journals = filtered_df['journal'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_journals)), top_journals.values, color='lightgreen')
    ax.set_yticks(range(len(top_journals)))
    ax.set_yticklabels(top_journals.index)
    ax.set_xlabel('Number of Publications')
    st.pyplot(fig)

with col4:
    st.subheader("Word Cloud of Titles")
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    all_titles = ' '.join(filtered_df['title'].apply(clean_text))
    if all_titles.strip():
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_titles)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No titles available for the selected filters.")

# Data table
st.subheader("Filtered Data")
st.dataframe(filtered_df[['title', 'authors', 'journal', 'year', 'publish_time']])
'''

# Save Streamlit app to a separate file
with open('cord19_app.py', 'w') as f:
    f.write(streamlit_code)

print("Streamlit app code saved to 'cord19_app.py'")
print("To run the Streamlit app, execute: streamlit run cord19_app.py")

# Part 5: Documentation and Reflection
print("\n--- DOCUMENTATION AND REFLECTION ---")

reflection = """
REFLECTION AND DOCUMENTATION:

1. DATA LOADING AND EXPLORATION:
   - Successfully loaded the CORD-19 metadata with {original_rows} rows and {original_cols} columns
   - Identified key columns for analysis: title, abstract, journal, publish_time
   - Found significant missing data in abstracts ({abstract_missing_pct:.1f}% missing)

2. DATA CLEANING CHALLENGES:
   - Had to handle multiple date formats in publish_time column
   - Made decision to remove papers without titles as they're essential for analysis
   - Chose to fill missing journal names with 'Unknown' rather than removing those rows

3. KEY FINDINGS:
   - Publication volume shows clear temporal patterns related to the pandemic
   - Certain journals dominate COVID-19 research output
   - Title analysis reveals common research themes and terminology

4. TECHNICAL LEARNING:
   - Gained experience with large dataset handling and cleaning
   - Practiced creating multiple visualization types
   - Learned to build interactive dashboards with Streamlit

5. CHALLENGES AND SOLUTIONS:
   - Memory issues with large dataset → used low_memory=False in pd.read_csv()
   - Inconsistent date formats → used errors='coerce' in pd.to_datetime()
   - Streamlit performance → implemented @st.cache_data for data loading

6. POTENTIAL IMPROVEMENTS:
   - Incorporate abstract text analysis for deeper insights
   - Add topic modeling to identify research themes
   - Include more interactive filters in the Streamlit app
   - Implement search functionality across titles and abstracts
""".format(
    original_rows=df.shape[0],
    original_cols=df.shape[1],
    abstract_missing_pct=(df['abstract'].isnull().sum() / len(df)) * 100
)

print(reflection)

# Save the cleaned dataset for future use
df_clean.to_csv('cord19_cleaned.csv', index=False)
print("\nCleaned dataset saved to 'cord19_cleaned.csv'")

print("\n=== ANALYSIS COMPLETE ===")
print("Summary of deliverables:")
print("1. ✓ Data loading and exploration completed")
print("2. ✓ Data cleaning and preparation finished") 
print("3. ✓ Analysis and visualizations generated")
print("4. ✓ Streamlit app created (cord19_app.py)")
print("5. ✓ Documentation and reflection written")
print("6. ✓ Cleaned dataset saved (cord19_cleaned.csv)")
