# DLVSML: Sentiment Analysis with Machine Learning and Deep Learning

## Overview
This project explores sentiment analysis on bilingual Malay–English datasets using both classical machine learning and deep learning approach.  
The goal is to compare traditional Machine Learning with transformer-based models (BERT) for English-Malay sentiments.

## Datasets
We combined multiple publicly available datasets:
- **Kaggle JSON Sentiment Data** (Positive/Negative tweets) https://www.kaggle.com/datasets/ilhamfp31/malaysia-twitter-sentiment
- **MESocSentiment Corpus** (Malay-English code-switched tweets) https://github.com/afifahms/MESocSentiment
- **News Sentiment Dataset** (Malay political/news texts) https://github.com/malaysia-ai/malaysian-dataset
- **Supervised Twitter Dataset** (Manually annotated tweets) https://github.com/malaysia-ai/malaysian-dataset
- **Supervised Twitter Politics Dataset** (Political tweets) https://github.com/malaysia-ai/malaysian-dataset
- **Annotated Bicodemix Dataset** (Malay-English bilingual sentiment + sarcasm labels) https://github.com/suhayryz/public_security_sa

All datasets were normalized into three sentiment classes:
- `0 = Positive`
- `1 = Negative`
- `2 = Neutral`

Dataset statistic summary are available in `data/dataset_summary.txt`.

## Methods
### Machine Learning Baselines
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  

Results are saved in `results/ml_results.txt`.

### Deep Learning
- Multilingual BERT (`bert-base-multilingual-cased`) fine-tuned on the unified dataset.  

Results are saved in `results/training_results.txt`.
