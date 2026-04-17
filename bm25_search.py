#-------------------------------------------------------------------------
# AUTHOR: Devaansh Mann
# FILENAME: bm25_search.py
# SPECIFICATION: BM25-based search engine with Average Precision evaluation
# FOR: CS 5180- Assignment #4
# TIME SPENT: ~2 hours
#-----------------------------------------------------------*/

# importing required libraries

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# Helper function: tokenize text and remove stopwords only
# ---------------------------------------------------------
def preprocess(text):

    tokens = text.lower().split()                                                  # 1. convert text to lowercase
    filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS]      # 2. remove stopwords only
    return filtered                                                                # 3. return the filtered tokens


# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# Files:
#   docs.csv
#   queries.csv
#   relevance_judgments.csv
# --> add your Python code here

docs_df = pd.read_csv("docs.csv")
queries_df = pd.read_csv("queries.csv")
relevance_df = pd.read_csv("relevance_judgments.csv")


# ---------------------------------------------------------
# 2. Build the BM25 index for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# Steps:
#   1. preprocess each document
#   2. store tokenized documents in a list
#   3. create the BM25 model
# --> add your Python code here

tokenized_docs = [preprocess(text) for text in docs_df["text"]]
doc_ids = docs_df["doc_id"].tolist()

# Create the BM25 model
bm25 = BM25Okapi(tokenized_docs)


# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# Suggested structure:
#   - for each query:
#       1. preprocess the query
#       2. compute BM25 scores for all documents
#       3. rank documents by score in descending order
#       4. retrieve the relevant documents for that query
#       5. compute AP
# --> add your Python code here

ap_scores = {}

for _, row in queries_df.iterrows():
    query_id = row["query_id"]
    query_text = row["query_text"]

    # 1. Preprocess the query
    tokenized_query = preprocess(query_text)

    # 2. Compute BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)

    # 3. Rank documents by score in descending order
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_doc_ids = [doc_ids[i] for i in ranked_indices]

    # 4. Retrieve the relevant documents for this query
    relevant_set = set(
        relevance_df[(relevance_df["query_id"] == query_id) & (relevance_df["judgment"] == "R")]["doc_id"])


    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    # Suggested steps:
    #   - initialize variables
    #   - go through the ranked documents
    #   - whenever a relevant document is found:
    #         precision = (# relevant found so far) / (current rank position)
    #         add precision to the running sum
    #   - AP = sum of precisions / total number of relevant documents
    #   - if there are no relevant documents, AP = 0

    # store the AP value for this query (use any data structure you prefer)
    num_relevant = len(relevant_set)

    if num_relevant == 0:
        ap_scores[query_id] = 0.0
        continue

    relevant_found = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_set:
            relevant_found += 1
            precision_sum += relevant_found / rank

    ap = precision_sum / num_relevant
    ap_scores[query_id] = ap


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here

sorted_queries = sorted(ap_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
# --> add your Python code here

print("====================================================")
print("Queries sorted by Average Precision (AP):")
print("====================================================")
for query_id, ap in sorted_queries:
    query_text = queries_df[queries_df["query_id"] == query_id]["query_text"].values[0]
    print(f"{query_id} ({query_text}): AP = {ap:.4f}")