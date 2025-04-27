import pandas as pd
import os
import json

# Base directory: from 'utils' folder → go up to project root
PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../."))
# Paths to data directories
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
REDDIT_DATA_DIR = os.path.join(RAW_DATA_DIR, "Reddit")
NEWS_HEADLINES_DIR = os.path.join(RAW_DATA_DIR, "News Headlines")
SARCASM_CORPUS_DIR = os.path.join(RAW_DATA_DIR, "Sarcasm Corpus v2")

# Paths to individual data files
REDDIT_DATA_PATH = os.path.join(REDDIT_DATA_DIR, "train-balanced-sarcasm.csv")
CORPUS_GEN_PATH = os.path.join(SARCASM_CORPUS_DIR, "GEN-sarc-notsarc.csv")
CORPUS_HYP_PATH = os.path.join(SARCASM_CORPUS_DIR, "HYP-sarc-notsarc.csv")
CORPUS_RQ_PATH = os.path.join(SARCASM_CORPUS_DIR, "RQ-sarc-notsarc.csv")
HEADLINES_JSON_PATH = os.path.join(
    NEWS_HEADLINES_DIR, "Sarcasm_Headlines_Dataset_v2.json")


def read_all_data_sources():
    # Read Reddit dataset
    reddit_data = pd.read_csv(REDDIT_DATA_PATH)
    reddit_data = reddit_data.rename(
        columns={"comment": "text", "label": "sarcastic"})
    reddit_data = reddit_data[["text", "sarcastic"]]
    reddit_data = reddit_data.dropna(subset=["text"])
    reddit_data = reddit_data[reddit_data["text"].str.strip() != ""]

    # Read JSON news headlines dataset

    headlines_data = pd.read_json(HEADLINES_JSON_PATH, lines=True)
    headlines_data = headlines_data.rename(
        columns={"headline": "text", "is_sarcastic": "sarcastic"})
    headlines_data = headlines_data[["text", "sarcastic"]]
    headlines_data = headlines_data.dropna(subset=["text"])
    headlines_data = headlines_data[headlines_data["text"].str.strip() != ""]

    # Read Sarcasm Corpus v2 datasets (CSV files)
    sarcasm_gen_data = pd.read_csv(CORPUS_GEN_PATH)
    sarcasm_hyp_data = pd.read_csv(CORPUS_HYP_PATH)
    sarcasm_rq_data = pd.read_csv(CORPUS_RQ_PATH)

    sarcasm_corpus_combined = pd.concat(
        [sarcasm_gen_data, sarcasm_hyp_data, sarcasm_rq_data])
    sarcasm_corpus_combined = sarcasm_corpus_combined.rename(
        columns={"class": "sarcastic"})
    sarcasm_corpus_combined = sarcasm_corpus_combined.replace(
        {"sarcastic": {"sarc": 1, "notsarc": 0}})
    sarcasm_corpus_combined = sarcasm_corpus_combined[["text", "sarcastic"]]
    sarcasm_corpus_combined = sarcasm_corpus_combined.dropna(subset=["text"])
    sarcasm_corpus_combined = sarcasm_corpus_combined[sarcasm_corpus_combined["text"].str.strip(
    ) != ""]
    twitter_headlines_data = pd.read_csv(
        rf"{DATA_DIR}/processed/twitter_headlines.csv").rename(
        columns={"text": "text", "class": "sarcastic"})
    twitter_headlines_data.replace(
        {"sarcastic": {"sarc": 1, "notsarc": 0}}, inplace=True)
    reddit_twitter_headlines_data = pd.concat([
        reddit_data,
        twitter_headlines_data
    ])
    # Combine all datasets
    combined_data = pd.concat([
        reddit_data,
        headlines_data,
        sarcasm_corpus_combined
    ], ignore_index=True)

    return reddit_data, headlines_data, sarcasm_corpus_combined, reddit_twitter_headlines_data, combined_data


if __name__ == "__main__":
    reddit_data, headlines_data, sarcasm_corpus_combined, reddit_twitter_headlines_data, combined_data = read_all_data_sources()
    print("Reddit Data:")
    print(reddit_data.shape)
    print("Headlines Data:")
    print(headlines_data.shape)
    print("Sarcasm Corpus Combined Data:")
    print(sarcasm_corpus_combined.shape)
    reddit_twitter_headlines_data.to_csv(
        os.path.join(DATA_DIR, "processed", "reddit_twitter_headlines.csv"), index=False)
    # sarcasm_corpus_combined.to_csv(
    #     os.path.join(DATA_DIR, "processed", "sarcasm_corpus_combined.csv"), index=False)
    # print(sarcasm_corpus_combined['sarcastic'].value_counts())

    # print(combined_data['sarcastic'].value_counts())
    # print(combined_data.shape)
    # print(combined_data.head())
    # # save to_csv_path = os.path.join(DATA_DIR, "raw", "combined_data.csv")
    # saveto_csv_path = os.path.join(
    #     DATA_DIR, "processed", "combined_data.csv")
    # # Save the combined data to a CSV file
    # combined_data.to_csv(saveto_csv_path, index=False)
