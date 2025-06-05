#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install beautifulsoup4 requests nltk openpyxl textblob')


# In[13]:


# importing the libraries
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


# In[14]:


#Uploading and Read Input.xlsx
input_df = pd.read_excel("Input.xlsx")
input_df.head()


# In[15]:


#Scrapping  Articles from URLs
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Title
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else ""

        # Article Body (may need adjustment per structure)
        content_div = soup.find('div', class_='td-post-content')
        article = content_div.get_text(separator=' ', strip=True) if content_div else ""

        return title + "\n" + article

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


# 

# In[16]:


#Saving Text Files with URL_ID
if not os.path.exists("articles"):
    os.makedirs("articles")

for _, row in input_df.iterrows():
    text = get_article_text(row['URL'])
    file_name = f"articles/{row['URL_ID']}.txt"
    with open(file_name, "w", encoding='utf-8') as f:
        f.write(text)


# In[19]:


# Loading Stopwords and Lexicons
def load_words_from_file(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        return set([line.strip().lower() for line in file if line.strip() and not line.startswith(";")])

stopword_files = [
    "StopWords_Auditor.txt", "StopWords_Currencies.txt", "StopWords_DatesandNumbers.txt",
    "StopWords_Generic.txt", "StopWords_GenericLong.txt", "StopWords_Names.txt"
]

stop_words = set()
for file in stopword_files:
    stop_words.update(load_words_from_file(file))

positive_words = load_words_from_file("positive-words.txt")
negative_words = load_words_from_file("negative-words.txt")


# In[20]:


#Cleaning Text
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text


# In[21]:


#Tokenizing Words and Sentences
from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_text(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences


# In[22]:


#Calculating Positive & Negative Score
def get_sentiment_scores(words):
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    return positive_score, negative_score


# In[23]:


#Calculating Polarity & Subjectivity Score
def get_polarity_score(pos, neg):
    return (pos - neg) / ((pos + neg) + 0.000001)

def get_subjectivity_score(pos, neg, total_words):
    return (pos + neg) / (total_words + 0.000001)


# In[24]:


#Complex Words, Syllables, and Fog Index
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiou"
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count

def is_complex(word):
    return syllable_count(word) > 2

def get_complex_word_metrics(words):
    complex_words = [word for word in words if is_complex(word)]
    percent_complex = len(complex_words) / len(words) if words else 0
    return len(complex_words), percent_complex


# In[25]:


#Readability Metrics (Fog Index, Sentence Lengths)
def get_readability_metrics(words, sentences, complex_word_count):
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
    fog_index = 0.4 * (avg_sentence_length + (complex_word_count / len(words)))
    return avg_sentence_length, avg_words_per_sentence, fog_index


# In[26]:


#Additional Metrics
def get_other_metrics(words, text):
    total_words = len(words)
    total_syllables = sum(syllable_count(word) for word in words)
    avg_word_length = sum(len(word) for word in words) / total_words if total_words else 0
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))
    return total_words, total_syllables / total_words, personal_pronouns, avg_word_length


# In[27]:


#Defining the Function to Analyze One Article
def analyze_article(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            text = f.read()

        cleaned = clean_text(text)
        words, sentences = tokenize_text(cleaned)

        # Remove stopwords
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]

        # Scores
        pos_score, neg_score = get_sentiment_scores(filtered_words)
        polarity = get_polarity_score(pos_score, neg_score)
        subjectivity = get_subjectivity_score(pos_score, neg_score, len(filtered_words))

        # Complexity
        complex_count, percent_complex = get_complex_word_metrics(filtered_words)

        # Readability
        avg_sent_len, avg_words_per_sent, fog_index = get_readability_metrics(filtered_words, sentences, complex_count)

        # Others
        word_count, syll_per_word, pronouns, avg_word_len = get_other_metrics(filtered_words, text)

        return [
            pos_score, neg_score, polarity, subjectivity,
            avg_sent_len, percent_complex, fog_index,
            avg_words_per_sent, complex_count, word_count,
            syll_per_word, pronouns, avg_word_len
        ]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [0]*13


# In[34]:


import nltk
nltk.download('punkt_tab')


# In[35]:


#Looping Through All Articles and Analyze
results = []

for _, row in input_df.iterrows():
    url_id = row['URL_ID']
    file_path = f"articles/{url_id}.txt"
    metrics = analyze_article(file_path)
    results.append(list(row) + metrics)  # Add URL_ID and URL from input row


# In[36]:


#Creating Final Output DataFrame
columns = list(input_df.columns) + [
    "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE",
    "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX",
    "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT",
    "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"
]

output_df = pd.DataFrame(results, columns=columns)


# In[37]:


#saving to excel file
output_df.to_excel("Output.xlsx", index=False)


# In[33]:


from google.colab import files
files.download("Output.xlsx")

