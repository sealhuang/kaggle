# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
import csv
from operator import itemgetter
import string

from patsy import dmatrices, dmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import classification_report
#joblib library for serialization
from sklearn.externals import joblib

def report(grid_scores, n_top=3):
    """
    Utility function to report best scores.

    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i+1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    score.mean_validation_score,
                    np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def substrings_in_string(big_string, substrings):
    """
    Utility to extract Title from Name.

    """
    for substring in substrings:
        if not string.find(big_string, substring) == -1:
            return substring
    print big_strting
    return np.nan

def replace_title_abbr(x):
    """
    Function for replacing all titles with Mr, Mrs, Miss, Master

    """
    title = x['Title']
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme', 'Mrs', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms', 'Miss']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def get_deck(x):
    loc_index = x['Cabin']
    loc_index = str(loc_index)
    if not loc_index == 'nan':
        if loc_index[0] == 'T':
            return 'U'
        else:
            return loc_index[0]
    else:
        return 'U'

def data_clean(df, test_df):
    """
    Clean data, imputation of missing values, and other feature enginering.

    """
    # create a title column from name
    title_list = ['Mrs', 'Mr', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme',
                  'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt',
                  'Countess', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df.apply(replace_title_abbr, axis=1)
    test_df['Title'] = test_df['Name'].map(lambda x: substrings_in_string(x, title_list))
    test_df['Title']=test_df.apply(replace_title_abbr, axis=1)

    # create new family_size column
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['HasFamily'] = df['Family_Size'] > 1
    test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch'] + 1
    test_df['HasFamily'] = test_df['Family_Size'] > 1

    # extract mean age for each group (Title x Pclass)
    mean_age = {}
    for t in df.Title[df.Age.isnull()].unique():
        mean_age[t] = {}
        for c in [1, 2, 3]:
            mean_age[t][c] = df.Age[(df.Age.notnull()) &
                                    (df.Title==t) &
                                    (df.Pclass==c)].mean()

    # refill NaN as mean age of each group
    df['AgeFill'] = df['Age']
    for t in df.Title[df.Age.isnull()].unique():
        for c in [1, 2, 3]:
            df.loc[(df.Title==t) &
                   (df.Age.isnull()) &
                   (df.Pclass==c), 'AgeFill'] = mean_age[t][c]

    test_df['AgeFill'] = test_df['Age']
    for t in test_df.Title[test_df.Age.isnull()].unique():
        for c in [1, 2, 3]:
            test_df.loc[(test_df.Title==t) &
                        (test_df.Age.isnull()) &
                        (test_df.Pclass==c), 'AgeFill'] = mean_age[t][c]

    # Age category
    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill<=10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill>60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill>10) & (df.AgeFill<=30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill>30) & (df.AgeFill<=60), 'AgeCat'] = 'senior'

    test_df['AgeCat'] = test_df['AgeFill']
    test_df.loc[(test_df.AgeFill<=10), 'AgeCat'] = 'child'
    test_df.loc[(test_df.AgeFill>60), 'AgeCat'] = 'aged'
    test_df.loc[(test_df.AgeFill>10) & (test_df.AgeFill<=30), 'AgeCat'] = 'adult'
    test_df.loc[(test_df.AgeFill>30) & (test_df.AgeFill<=60), 'AgeCat'] = 'senior'

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if df.Embarked[df.Embarked.isnull()].size:
        df.loc[df.Embarked.isnull(),
               'Embarked'] = df.Embarked.dropna().mode().values
    
    if test_df.Embarked[test_df.Embarked.isnull()].size:
        test_df.loc[test_df.Embarked.isnull(),
                    'Embarked'] = test_df.Embarked.dropna().mode().values
    
    # Special case for cabins as NaN may be signal
    df['Deck']=df.apply(get_deck, axis=1)
    print 'Unique deck locations (Training):'
    print df.Deck.unique()

    test_df['Deck']=test_df.apply(get_deck, axis=1)
    print 'Unique deck locations (Testing):'
    print test_df.Deck.unique()
    
    # we set those fares of 0 to NaN
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    test_df.Fare = test_df.Fare.map(lambda x: np.nan if x==0 else x)

    # extract mean fare from each group (Pclass)
    for c in [1, 2, 3]:
        df.loc[(df.Fare.isnull()) & (df.Pclass==c), 'Fare'] = np.median(df[df.Pclass==c]['Fare'].dropna())
        test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass==c), 'Fare'] = np.median(df[df.Pclass==c]['Fare'].dropna())

    # ad hoc features
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
    df['AgeClass'] = df['AgeFill'] * df['Pclass']
    df['ClassFare'] = df['Pclass'] * df['Fare_Per_Person']
    test_df['Fare_Per_Person'] = test_df['Fare'] / test_df['Family_Size']
    test_df['AgeClass'] = test_df['AgeFill'] * test_df['Pclass']
    test_df['ClassFare'] = test_df['Pclass'] * test_df['Fare_Per_Person']
    
    df['FareLevel'] = df['Pclass']
    df.loc[(df.Fare<55), 'FareLevel'] = 'Low'
    df.loc[(df.Fare>=55) & (df.Fare<155), 'FareLevel'] = 'Med'
    df.loc[(df.Fare>=155), 'FareLevel'] = 'High'
    test_df['FareLevel'] = test_df['Pclass']
    test_df.loc[(test_df.Fare<55), 'FareLevel'] = 'Low'
    test_df.loc[(test_df.Fare>=55) & (test_df.Fare<155), 'FareLevel'] = 'Med'
    test_df.loc[(test_df.Fare>=155), 'FareLevel'] = 'High'

    # encoding category variable
    le = preprocessing.LabelEncoder()
    enc = preprocessing.OneHotEncoder()

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)
    x_sex = le.transform(test_df['Sex'])
    test_df['Sex'] = x_sex.astype(np.float)
    
    le.fit(df['Title'])
    x_title = le.transform(df['Title'])
    df['Title'] = x_title.astype(np.float)
    x_title = le.transform(test_df['Title'])
    test_df['Title'] = x_title.astype(np.float)

    le.fit(df['Deck'])
    x_loc = le.transform(df['Deck'])
    df['Deck'] = x_loc.astype(np.float)
    x_loc = le.transform(test_df['Deck'])
    test_df['Deck'] = x_loc.astype(np.float)

    le.fit(df['HasFamily'])
    x_hl = le.transform(df['HasFamily'])
    df['HasFamily'] = x_hl.astype(np.float)
    x_hl = le.transform(test_df['HasFamily'])
    test_df['HasFamily'] = x_hl.astype(np.float)

    le.fit(df['FareLevel'])
    x_hl = le.transform(df['FareLevel'])
    df['FareLevel'] = x_hl.astype(np.float)
    x_hl = le.transform(test_df['FareLevel'])
    test_df['FareLevel'] = x_hl.astype(np.float)

    le.fit(df['AgeCat'])
    x_age = le.transform(df['AgeCat'])
    df['AgeCat'] = x_age.astype(np.float)
    x_age = le.transform(test_df['AgeCat'])
    test_df['AgeCat'] = x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb = le.transform(df['Embarked'])
    df['Embarked'] = x_emb.astype(np.float)
    x_emb = le.transform(test_df['Embarked'])
    test_df['Embarked'] = x_emb.astype(np.float)

    test_df['Survived'] =  [0 for x in range(len(test_df))]
    
    # Remove the useless column
    df = df.drop(['Name', 'Age', 'Ticket', 'PassengerId'], axis=1) 
    test_ids = test_df['PassengerId'].values
    test_df = test_df.drop(['Name', 'Age', 'Ticket', 'PassengerId'], axis=1) 
    
    return df, test_ids, test_df

def model_training():
    """
    Function for model training.

    """
    seed = 0
    # load data
    train_df = pd.read_csv('train.csv', header=0)
    test_df = pd.read_csv('test.csv', header=0)
    train_data, test_ids, test_data = data_clean(train_df, test_df)

    # formula
    formula_ml = 'Survived ~ Pclass + C(Title) + Sex + AgeFill + Fare + Family_Size + C(Deck) + C(Embarked) + C(FareLevel) + SibSp + Parch + HasFamily'

    y_train, x_train = dmatrices(formula_ml, data=train_data, return_type='dataframe')
    y_train = np.asarray(y_train).ravel()
    print y_train.shape, x_train.shape

    y_test, x_test = dmatrices(formula_ml, data=test_data, return_type='dataframe')
    y_test = np.asarray(y_test).ravel()
    print y_test.shape, x_test.shape

    #forest = RandomForestClassifier(n_estimators=300, criterion='entropy',
    #                                oob_score=True, max_depth=10,
    #                                random_state=seed, min_samples_split=1,
    #                                min_samples_leaf=1, n_jobs=1)
    #forest = forest.fit(x_train, y_train)
    #print 'OOB score: ',
    #print forest.oob_score_

    #print 'Predicting...'
    #output = forest.predict(x_test).astype(int)

    #predictions_file = open("myfirstresult.csv", "wb")
    #open_file_object = csv.writer(predictions_file)
    #open_file_object.writerow(["PassengerId", "Survived"])
    #open_file_object.writerows(zip(test_ids, output))
    #predictions_file.close()
    #print 'Done.'

    #-- model selection
    # select a train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train,
                                                   test_size=0.2,
                                                   random_state=seed)

    # instantiate and fit model
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                 max_depth=5, min_samples_split=1,
                                 min_samples_leaf=1, max_features='auto',
                                 bootstrap=False, oob_score=False,
                                 n_jobs=1, random_state=seed,
                                 verbose=0, min_density=None,
                                 compute_importances=None)

    # conduct grid search to find best parameters for pipeline
    param_grid = {'clf__n_estimators': [100, 150, 200, 250, 300],
                  'clf__max_depth': [1, 3, 5, 10, 15],
                  'clf__min_samples_leaf': [1, 3, 5, 7, 9, 11, 13, 15, 17]}
    pipeline = Pipeline([('clf', clf)])
    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               verbose=0, scoring='accuracy',
                               cv=StratifiedShuffleSplit(Y_train, n_iter=10,
                                            test_size=0.2, train_size=None,
                                            indices=None, random_state=seed,
                                            n_iterations=None)).fit(
                            X_train, Y_train)
    # score the results
    print('Best score: %0.3f'%grid_search.best_score_)
    print(grid_search.best_estimator_)
    report(grid_search.grid_scores_)

    print('-----grid search end------------')
    print ('on all set')
    scores = cross_val_score(grid_search.best_estimator_,
                             x_train, y_train, cv=3, scoring='accuracy')
    print scores.mean(), scores
    print ('on test set')
    scores = cross_val_score(grid_search.best_estimator_,
                             X_test, Y_test, cv=3, scoring='accuracy')
    print scores.mean(), scores

    print ('train set')
    print(classification_report(Y_train,
                                grid_search.best_estimator_.predict(X_train)))
    print('test data')
    print(classification_report(Y_test,
                                grid_search.best_estimator_.predict(X_test)))

    print 'Predicting...'
    output = grid_search.best_estimator_.predict(x_test).astype(int)
    predictions_file = open("result.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(test_ids, output))
    predictions_file.close()
    print 'Done.'

if __name__ == '__main__':
    model_training()


