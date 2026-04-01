# DLVSML: Sentiment Analysis with Machine Learning and Deep Learning

## Overview
This project explores sentiment analysis on bilingual Malay–English datasets using both classical machine learning and deep learning approach.  
The goal is to compare traditional Machine Learning with transformer-based models (BERT) for English-Malay sentiments.

## Datasets
We combined multiple publicly available datasets:
- **Kaggle JSON Sentiment Data** https://www.kaggle.com/datasets/ilhamfp31/malaysia-
- **Kaggle The dUCk Tweets** https://www.kaggle.com/datasets/carlsonhoo/the-duck-tweets
- **MESocSentiment Corpus** https://github.com/afifahms/MESocSentiment
- **Covid Malaysia** https://github.com/sarahjuan/malaysia-tweets-with-sentiment-labels
- **Semi-Supervised-Twitter** https://malaysian-dataset.readthedocs.io/en/latest/sentiment.html
- **Annotated Bicodemix Dataset** https://github.com/suhayryz/public_security_sa

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
