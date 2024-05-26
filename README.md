
# NLP and Naive Bayes Spam Detection Model

This project demonstrates how to build a spam detection system using Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes algorithm. The system can classify SMS messages as spam or not spam with a high degree of accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Results](#results)

## Introduction

Spam detection is a crucial task in modern communication systems. The aim of this project is to build a machine learning model to identify spam messages from a collection of SMS messages using NLP and Naive Bayes classifier.

## Dataset

The dataset used is the 'SMSSpamCollection' dataset, which contains a collection of SMS labeled messages. Each message is labeled as either 'ham' (non-spam) or 'spam'.

## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn

You can install the necessary packages using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

## Project Structure

```
├── README.md
├── nlp_spam_detection.py
├── naive_bayes_spam_detection.py
├── SMSSpamCollection.txt
```

- `nlp_spam_detection.py`: Script for NLP spam detection with preprocessing steps.
- `naive_bayes_spam_detection.py`: Script for spam detection using Multinomial Naive Bayes.
- `SMSSpamCollection.txt`: Dataset file.

## Results

The results include the accuracy, precision, recall, F1 score, and a confusion matrix of the spam detection model. The model can also predict whether a new SMS message is spam or not, with a probability score.

