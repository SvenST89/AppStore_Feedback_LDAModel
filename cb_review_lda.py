# -*-coding:utf-8 -*-
'''
@File    :   cb_review_lda.py
@Time    :   2024/10/08 21:18:07
@Author  :   Sven STEINBAUER
@Version :   1.0
@Contact :   svensteinbauer89@googlemail.com
@License :   (C)Copyright 2024, Sven STEINBAUER
@Desc    :   None
'''

import logging
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import pickle
import json
from typing import List
import asyncio
# OWN LIBS
from get_reviews import GetReviews
# LDA
from gensim.models import Phrases
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA

# LANGUAGE MODELLING: tokenization, stopword and punctuation removal, lemmatization und word type choice
import nltk
import spacy
#from spacy.lang.de.stop_words import STOP_WORDS
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# # NLTK ---- packages to load the first time you execute the script -----------------------
# # Download NLTK stopwords for German
# nltk.download('stopwords')
# # punkt is used under the hood, once loaded, by the function `word_tokenize()` of nltk
# nltk.download('punkt')
# # for lemmatization with nltk
# nltk.download('wordnet')
#------------------------------------------------------------------------------------------

# VISUALIZATION
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvisualize
# from IPython.display import display, Markdown
# Visualizes in 3D with plotly-express to make it interactive
import plotly.express as px

#========================================#
# Preparation
#========================================#
PROJECT_ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, r"data")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, r"model\outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

#========================================#
# LOGGING
#========================================#
# the general log-level logs anything that is above the given level, e.g., INFO logs Warnings, Errors and Critical messages, but not DEBUG.
# If we do not set a log-level at a specific Handler, this handler inherits the level from the general logger-level.
logger = logging.getLogger(__name__)
# set general log-level
logger.setLevel("INFO")
formatter = logging.Formatter('{asctime} - {levelname} - {lineno} - {message}', style="{", datefmt="%Y-%m-%d %H:%M",)

# set logging handler in file
fileHandler = logging.FileHandler(filename=fr"{LOG_DIR}\lda.log", mode="a", encoding="utf-8")
fileHandler.setLevel("INFO")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# set logging handler in console output
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

#========================================#
# VARIABLES
#========================================#
app_name="consorsbank"
app_id="930883278"
rcount = 150
passes = 200
iterations = 200

#========================================#
# FUNCTIONS & MAIN CODE
#========================================#
# Load csv and write each review with its content into an extra txt-file as a document
#df = pd.read_csv(filepath_or_buffer="data/consorsbank_reviews.csv")
# Find app ID from Apple AppStore: https://apps.apple.com/de/app/consorsbank/id930883278
def get_data(aname, aid, r_count):    
    cb = GetReviews(appname=aname, appid=aid, review_count=r_count)
    return cb.scrape_and_store_reviews()

# customize stopword list
def cust_stopwords(custom_stopwordlist: List):
    global stopwords_ger
    if not isinstance(custom_stopwordlist, List):
        raise TypeError("Your passed stopword list ist not a list of strings!")
    stopwords_ger = stopwords.words("german")
    stopwords_ger.extend(custom_stopwordlist)

# Pack reviews into a list
def review_list(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Your passed data is not a dataframe!")
    german_feedback=[]
    for idx, row in df.iterrows():
        # populate list with feedback
        german_feedback.append(str(row['review']))
    return german_feedback

def write_files(processed_feedback: List[List[str]]):
    # To check all the lemmatized words, write them into a txt file
    with open("words.txt", "w", encoding="utf-8") as f:
        for t in range(0,len(processed_feedback)):
            wordlist = processed_feedback[t]
            for w in wordlist:
                f.write(w + "\n")

    with open("words.txt", "r", encoding="utf-8") as r:
        lines = r.readlines()
        words = []
        for l in lines:
            words.append(l.replace("\n", "").lstrip().rstrip())
        vocab_size = len(words)
    
    # Set vocab size as environment variable
    os.environ['VOCAB_SIZE']=str(vocab_size)

# 4. Function to compute coherence score for an LDA model
# If u_mass as coherence metric, corpus should be provided: https://radimrehurek.com/gensim/models/coherencemodel.html
# u_mass better than c_v: https://www.baeldung.com/cs/topic-modeling-coherence-score
async def compute_coherence(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='u_mass')
    return coherence_model.get_coherence()
    
# Text Preprocessing Function (For German)
def preprocess_text(texts):
    # SPACY
    # for tokenization and lemmatization (grouping inflected forms of a word, e.g. break, broken, breaks go all into one group)
    nlp = spacy.load('de_core_news_sm')
    custom_stopwords = ["mal", "bitte", "danke", "nem", "leider", "app", "consors", "consorsbank", "bank", "handy", "apps", "bzw", "funktionen"]
    cust_stopwords(custom_stopwordlist=custom_stopwords)
    positive_wordset = {'ADJ', 'NOUN', 'PROPN'}
    unique_noun_set = set()
    # Use German stopwords
    stop_words = set(stopwords_ger)
    processed_texts = []

    for text in texts:
        # Tokenize and convert to lowercase, use punctuation in the background of nltk-word_tokenize to segment text into words or sentences
        tokens = word_tokenize(text.lower())
        # Remove once more punctuation stopwords
        tokens = [word.lower() for word in tokens if word not in string.punctuation and word not in stop_words]
        doc = nlp(" ".join(tokens))
        words = [str(token) for token in doc if token.pos_ in positive_wordset]
        # lemmatize words to word root
        lemmatizer = WordNetLemmatizer()
        doc_i = [lemmatizer.lemmatize(w) for w in words]
        nouns = [str(w) for w in doc if w.pos_ in {'NOUN', 'PROPN'}]

        for v in nouns:
            if v not in unique_noun_set:
                unique_noun_set.add(v)
        # Keep words with more than 2 characters and remove stopwords again after lemmatization
        processed_texts.append([word.lower() for word in doc_i if word.isalpha() and len(word) >= 3])
        
    logger.info(f"Documents successfully preprocessed. Length of docs are: {len(processed_texts)}.")
    
    # now set vocab size env-variable and write files for checking
    write_files(processed_feedback=processed_texts)

    return processed_texts, unique_noun_set

def prepare_data(dataframe):
    german_feedback = review_list(df=dataframe)
    logger.info(f"Feedback is of length: {len(german_feedback)}.")
    processed_texts, unique_noun_set = preprocess_text(german_feedback)
    # Create bigrams
    # min_count:  Ignore all words and bigrams with total collected count lower than this value
    docs = deepcopy(processed_texts)
    vocab_size = os.environ['VOCAB_SIZE']
    bigram = Phrases(docs, min_count=20, max_vocab_size=int(vocab_size))
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Create a Dictionary and Corpus for LDA
    # Dictionary: Mapping of words to an id
    dictionary_german = corpora.Dictionary(docs)
    # Filter extremes to remove too rare words (not below 2 occurrences) and too frequent words appearing in more than x% of documents
    dictionary_german.filter_extremes(no_below=2, no_above=0.25)
    # Corpus: Bag of words representation of the texts
    corpus_german = [dictionary_german.doc2bow(text) for text in docs]
    logger.info(f"Length of corpus is: {len(corpus_german)}")
    
    return docs, dictionary_german, corpus_german

#@log(my_logger=MyLogger())
async def train_and_tune_lda(documents, dictionary, corpora):
    # Grid search for hyperparameter tuning
    param_grid = {
        'num_topics': [10, 15, 17, 20, 22, 25],  # Try different topic numbers
        'alpha': ['symmetric', 'asymmetric', 0.01, 0.1],  # Alpha values
        'eta': ['symmetric', 0.01, 0.1]  # Eta (beta) values
    }
    
    #----------------------------------------------#
    # START TRAINING / TUNING
    #----------------------------------------------#
    best_model = None
    best_coherence = -1
    best_params = {}
    rounds = 1
    # Iterate over the grid of hyperparameters
    for params in ParameterGrid(param_grid):
        logger.info("")
        logger.info(f"\n--------------------------------------------------------\nROUND {rounds}\n--------------------------------------------------------\n")
        logger.info(f"Testing with params: {params}")
        # Build the LDA model
        # increase passes and iterations to learn more from the data
        # check this for info on saving, loading and retraining with new data: https://radimrehurek.com/gensim/models/ldamodel.html
        lda_model = LdaModel(corpus=corpora,
                            id2word=dictionary,
                            num_topics=params['num_topics'],
                            random_state=42,
                            chunksize = 500,
                            alpha=params['alpha'],
                            eta=params['eta'],
                            passes=passes,
                            iterations=iterations)
        
        # Compute the coherence score
        coherence = await compute_coherence(lda_model, documents, dictionary)
        logger.info(f"Coherence Score: {coherence}")
        logger.info("")

        # Track the best model based on coherence
        if coherence > best_coherence:
            best_model = lda_model
            pickle.dump(best_model, open(rf"{MODEL_DIR}\best_lda_model.pkl", "wb")) # then to load again: lda_mod = pickle.load(open("best_lda_model.pkl", "rb"))
            best_coherence = coherence
            best_params = params
            best_params["best_coherence_score"] = coherence
            with open(rf"{MODEL_DIR}\best_params.json", "w") as f:
                json.dump(best_params, f)
        
        rounds += 1
    
    # 6. Output the best model and its coherence score
    logger.info(f"\nBest Parameters: {best_params}")
    logger.info(f"Best Coherence Score: {best_coherence}")

    # 7. Visualize the topics from the best model
    topics = best_model.print_topics(num_words=10)
    for topic in topics:
        logger.info(f"\nTopic {topic[0]}: {topic[1]}")
        
    return best_model, best_params, best_coherence

def make_pca(df, best_model, corpus):
    # Extract topic distributions for each feedback
    topic_dists = [best_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    # convert the distribution list into a matrix where each row corresponds to the topic distribution of each feedback
    topic_matrix = np.array([[prob for _, prob in doc] for doc in topic_dists])
    
    # Dimensionality reduction with PCA (reduce to 3 dimensions)
    pca = PCA(n_components=3, random_state=42)
    topic_pca = pca.fit_transform(topic_matrix)
    
    top_topics = best_model.top_topics(corpus)
    topic_collection ={}
    for j in range(0, len(top_topics)):
        word_list = []
        k = 0
        while k < 10:
            
            #print(top_topics[0][0][k][1])
            word_list.append(top_topics[j][0][k][1])
            k+=1
        topic_collection[f"Topic {j+1}"]= "; ".join(word_list)
    with open(rf"{MODEL_DIR}\topic_collection.json", "w") as t:
        json.dump(topic_collection, t, ensure_ascii=False)
        
    # Make dataframe of my data
    # topic_pca delivers for each of the 1500 feedbacks an array of three values which you can access
    # by topic_pca[i]
    dict_list=[]
    topic_dict_keys = list(topic_collection.keys())
    for l in range(0, len(topic_pca)):
        df_dict = {}
        df_dict['x'] = topic_pca[l][0]
        df_dict['y'] = topic_pca[l][1]
        df_dict['z'] = topic_pca[l][2]
        # map the related topic with the help of pd.Series.idxmax() and the topic_matrix above
        max_dist_value_topic_idx = pd.Series(topic_matrix[l]).idxmax()
        df_dict['topic'] = topic_dict_keys[max_dist_value_topic_idx]
        df_dict['related_words'] = topic_collection[topic_dict_keys[max_dist_value_topic_idx]]
        df_dict['feedback_given'] = df.loc[l, "review"]
        dict_list.append(df_dict)
    pca_df = pd.DataFrame(dict_list)
    pca_df.to_csv(rf"{MODEL_DIR}\lda_pca_df.csv", header=True)
    
    return pca_df

def make_graph(pca_df):
    fig = px.scatter_3d(pca_df, x='x', y='y', z='z',
            color='topic')
    fig.show()
    
    return True

async def main() -> None:

    df = get_data(aname=app_name, aid=app_id, r_count=rcount)
    docs, dictionary_german, corpus_german = prepare_data(dataframe=df)
    best_model, best_params, best_coherence = await train_and_tune_lda(docs, dictionary_german, corpus_german)
    pca_df = make_pca(df, best_model, corpus_german)
    make_graph(pca_df=pca_df)

if __name__ == "__main__":
    asyncio.run(main())