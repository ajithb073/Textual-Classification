## Description
 Through Machine Learning and NLP we are classifying text comprising of keywords "Thank you for applying" into : applied for Job{array([1])} and not applied for Job{array([0])} in Python.

## Requirements

1. ipython notebook - Python Text Editor.
2. Pandas  - library used for data analysis.
3. Spacy  - nlp library in python.
4. Sklearn - useful library for creating ML models.

## Instructions to Code execution

1. Download the dataset from link below and place it in the working directory.
2. Name specified as "Code" below must be executed in ipython notebook 

## Source data
 
  Data set : https://docs.google.com/spreadsheets/d/1BsefOYmrz59T3fHUXe27X571XEMLrQc5Yje8rG3jxQA/edit?usp=sharing
 
# import pandas and dataframe creation
  Pandas are easy-to-use data structures and data analysis tools for the Python programming language.

Code: import pandas as pd
      df = pd.read_csv("abcd - data.csv")

# Converting column[Job_Application] from categorical to numerical value and creating an new column[label]
Code: df['label'] = df['Job_Application'].apply(lambda x: 0 if x=='No' else 1)

# View of dataframe head 
Code: df.head()

## Install Spacy in ipython-notebook
  spaCy is an open-source natural language processing library for Python.
 # Installing Spacy - Linux
Code:  !pip install spacy
       !python -m spacy download en
       import spacy

## Importing Libraries

  Count Vectorization involves counting the number of occurrences each words appears in a document.
  TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator.
  
Code: from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
      from sklearn.base import TransformerMixin
      from sklearn.pipeline import Pipeline

## Tokening the Data With spaCy

Python’s string module, which contains a helpful list of all punctuation marks that we can use in string.punctuation.
Then, we’ll create a spacy_tokenizer() function that accepts a sentence as input and processes the sentence into tokens, performing lemmatization, lowercasing, and removing stop words

Code: import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
    # Create our list of punctuation marks
punctuations = string.punctuation
    # Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
    # Load English tokenizer, tagger, parser, NER and word vectors
parser = English()
    # Creating our tokenizer function
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    # Removing stop words
mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    # return preprocessed list of tokens
 return mytokens

## Defining a Custom Transformer
create a custom transformer for removing initial and end spaces and converting text into lower case.
We create a custom predictors class which inherits the TransformerMixin class. This class overrides the transform, fit and get_parrams methods.
create a clean_text() function that removes spaces and converts text into lowercase.

Code:
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
 def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

## Vectorization Feature Engineering (TF-IDF)

BoW converts text into the matrix of occurrence of words within a given document.

Code:
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

## Splitting The Data into Training and Test Sets

scikit-learn gives us a built-in function for doing this: train_test_split(). 

Code:
from sklearn.model_selection import train_test_split
X = df['Email_Subject'] # the features we want to analyze
ylabels = df['label'] # the labels, or answers, we want to test against
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

## Creating a Pipeline and Generating the Model
 We will import the LogisticRegression module and create a LogisticRegression classifier object.

Then, we’ll create a pipeline with three components: a cleaner, a vectorizer, and a classifier. The cleaner uses our predictors class object to clean and preprocess the text. The vectorizer uses countvector objects to create the bag of words matrix for our text. The classifier is an object that performs the logistic regression to classify the sentiments.

Once this pipeline is built, we’ll fit the pipeline components using fit().

Code:
 # Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
 # Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])
 # model generation
pipe.fit(X_train,y_train)

## Evaluating the Model
We are using the metrics module from scikit-learn. 

Accuracy refers to the percentage of the total predictions our model makes that are completely correct.
Precision describes the ratio of true positives to true positives plus false positives in our predictions.
Recall describes the ratio of true positives to true positives plus false negatives in our predictions.

Code:
from sklearn import metrics
 # Predicting with a test dataset
predicted = pipe.predict(X_test)
 # Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

 # Output
 Logistic Regression Accuracy: 0.9375
 Logistic Regression Precision: 1.0
 Logistic Regression Recall: 0.8888888888888888

## Real-Time Prediction

 For a given sentence, we will predict whether applied for job{array([1])}
  or not applied for job{array([0])}

1. pipe.predict(["Thank you for applying for the role of Software Developer"])
   Output = array([1])

2. pipe.predict(["Please do review my application"]) 
   Output = array([0])

## Future Improvements

1. By increasing the number of Independent variables, it can lead to better evaluation of Keywords.
2. If Model performance is increased, it thus creates a better classification model.




