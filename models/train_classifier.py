import pickle
import sys

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load X and y from the database in the recieved path.

    Arguments:
        database_filepath {String} -- Path of sqlite database.

    Returns:
        X {pandas series} -- Message
        y {pandas framework} -- encoded labels
        categories {list} -- categories list
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1) 
    categories = category_names = y.columns.tolist()
    return X, y, categories


def tokenize(text):
    """Tokenize the recieved text.
    
    Arguments:
        text {String} -- Disaster message
    
    Returns:
        list -- tokens
    """    
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    lem_tokens = []
    for t in tokens:
        lem_tokens.append(lemmatizer.lemmatize(t.lower()))
    return lem_tokens


def build_model():
    """Build a classifer and optimize it with grid seach.
    
    Returns:
        GridSearchCV -- sklearn grid search model object.
    """    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__min_samples_split': [3, 5, 8],
                    'clf__estimator__max_depth': [3, 8]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=1)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset.
    
    Arguments:
        model {sklearn model} -- trained model  
        X_test {pandas series} -- test set
        Y_test {pandas framework} -- test set labels
        category_names {list} -- category names
    """    
    pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print("Column: {}".format(col))
        print(classification_report(Y_test[col], pred[:, i]))


def save_model(model, model_filepath):
    """Save model as pickled file
    
    Arguments:
        model {sklearn object} -- trained model
        model_filepath {String} -- path to store the model
    """    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
