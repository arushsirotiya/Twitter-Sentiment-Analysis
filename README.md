# Twitter-Sentiment-Analysis

# Overview
This project is a sentiment analysis model trained to classify tweets as either positive or negative. The model uses logistic regression on text data that has been preprocessed using stemming and TF-IDF vectorization. The dataset used for training contains 1.6 million tweets labeled with sentiment (0 for negative, 4 converted to 1 for positive).

# Key Features
- Data preprocessing including:
  - Text cleaning
  - Stemming using Porter Stemmer
  - Stopword removal
- Feature extraction using TF-IDF vectorization
- Logistic Regression model with ~80% accuracy
- Model persistence using pickle for future predictions

# Dataset
The dataset used is "Sentiment140" from Kaggle, containing:
- 1.6 million tweets
- Balanced classes (800k positive, 800k negative)
- Original columns: target, id, date, flag, user, text

# Model Performance
- Training accuracy: 79.87%
- Testing accuracy: 79.87%

# Requirements
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - nltk
  - scikit-learn
  - pickle

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook to:
   - Preprocess the data
   - Train the model
   - Evaluate performance
   - Save the trained model

# Files
- `trained_model.sav`: Serialized model for predictions
- Jupyter notebook containing all code

# How to Make Predictions
```python
import pickle

# Load the model
model = pickle.load(open('trained_model.sav', 'rb'))

# Preprocess new text (apply same stemming/cleaning)
processed_text = preprocess_function("Your tweet text here")

# Vectorize using the same TF-IDF vectorizer
vectorized_text = vectorizer.transform([processed_text])

# Make prediction
prediction = model.predict(vectorized_text)

if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
```

# Future Improvements
- Try different classification algorithms
- Experiment with deep learning approaches
- Add more sophisticated text preprocessing
- Include emoji/emoticon analysis
- Deploy as a web service

# Acknowledgements
Dataset from Kaggle: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
