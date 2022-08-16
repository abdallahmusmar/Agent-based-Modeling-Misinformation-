import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import random
from random import shuffle, randint
from tqdm import tqdm_notebook as tqdm
import networkx as nx
from pprint import pprint
import pickle
import seaborn as sns 
from SimHelper import *

#distributions
from scipy.stats import bernoulli, norm, truncnorm, expon, gamma, beta

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

#model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix,accuracy_score

from pipelinehelper import PipelineHelper 

pipe = Pipeline([
    ('scaler', PipelineHelper([
        ('std', StandardScaler()),
        ('max', MaxAbsScaler()),
    ])),
    ('classifier', PipelineHelper([
        ('svm', SVC()),
        ('rf', RandomForestClassifier()),
        ('knn',KNeighborsClassifier()),
        ('gnb',GaussianNB()),
        ('log',LogisticRegression()),
    ])),
])

params = {
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'std__with_mean': [True, False],
        'std__with_std': [True, False],
        'max__copy': [True],  # just for displaying
    }),
    'classifier__selected_model': pipe.named_steps['classifier'].generate({
        'svm__C': [0.1, 1.0],
        'svm__gamma': ['auto','scale'],
        'rf__n_estimators': [100, 20],
        'knn__weights':['uniform','distance'],
        'knn__n_neighbors':[3,5],
    })
}

class NewsArticle:
    """This is a news article that has a title sentiment and a number of shares"""
    def __init__(self,news_article_id,source_id=0,source_bias=0,sentiment=0,num_shares=0,fake=0):
        self.news_article_id = news_article_id
        self.source_id = source_id
        self.source_bias=source_bias
        self.sentiment=sentiment #value between (-5,-1) and (1,5)
        self.num_shares = num_shares # any value >=0
        self.fake=fake #either real=0 or fake=1
		
class NewsSource:
    """This is a news source that publish-es articles"""
    def __init__(self,news_source_id,bias=0,reliable=0,news_articles=[]):
        self.news_source_id = news_source_id
        self.bias=bias #how much bias a source has towards the topic=1 or againts the topic=-1
        self.reliable=reliable #whether the news source reliable=1 or not=0
        self.news_articles = [] #list of NewsArticle(s) that the source published
    
    def add_article(self,news_article):
        """add a NewsArticle to the list of articles"""
        self.news_articles.append(news_article)
#         if self.news_source_id == news_article.source_id:
#             self.news_articles.append(news_article)
#         else:
#             print('News article belongs to another source')

# RandomForestClassifier(n_estimators =100,n_jobs=2, min_samples_leaf=2,random_state=0)
class UserAgent:
    def __init__(self,user_id,bias=0,analytical=0,source_credibility={},news_articles=[],classifier=GridSearchCV(pipe, params, scoring='accuracy', verbose=1, cv=4, n_jobs=-1)):
        self.user_id = user_id
        self.bias=bias #how much bias a user has towards the topic=1 or againts the topic=-1
        self.analytical=analytical #whether the user judges articles based on analysis=1 or not=0
        self.source_credibility=source_credibility #the perceived credibility for each source
        self.news_articles = [] #list of NewsArticle(s) that the user received
        self.classifier = classifier
        
    def add_article(self,news_article):
        """add a NewsArticle to the list of articles"""
        self.news_articles.append(news_article)

    def generate_source_credibility(self,news_sources):
        """
        assign source credibility scores given a user's bias
        the higher the bias,the more extreme the credibility scores 
        """
        if len(self.source_credibility)==0:
            sources_set = set()
            for news_article in self.news_articles:
                sources_set.add(news_article.source_id)
            self.source_credibility = dict.fromkeys(sources_set,0)
            for source in self.source_credibility:
#                 self.source_credibility[source] = round(1-(abs(news_sources[source].bias - self.bias))/2,2)
                similarity_in_bias = abs(news_sources[source].bias - self.bias)
#                 print(similarity_in_bias)
                similarity_in_bias = similarity_in_bias/2
#                 print(similarity_in_bias)
                similarity_in_bias = 1- similarity_in_bias
#                 print(similarity_in_bias)
                mean_shift = 0.2
                similarity_in_bias = abs(round(similarity_in_bias-abs(self.bias - mean_shift),2))
            
                self.source_credibility[source] = generate_weighted_percentage(similarity_in_bias)
                print('source:',source,',similarity:',similarity_in_bias,'credibility:',self.source_credibility[source])


    def train_classifier(self):
        """trains news articles where time has passed and are already known to be either fake or real"""
        df = pd.DataFrame(columns=['analytical','source_credibility','sentiment','num_shares','fake'])
        print('populating the dataframe')
        for news_article in self.news_articles:
            df = df.append(pd.Series([self.analytical, self.source_credibility[news_article.source_id],news_article.sentiment, news_article.num_shares, news_article.fake], index=df.columns) , ignore_index=True)
        print(df.tail())
        
        # Labels are the values we want to predict
        DV = np.array(df['fake'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        df = df.drop('fake', axis = 1)
        # Saving feature names for later use
        feature_list = list(df.columns)
        
        # Convert to numpy array
        IVs = np.array(df)
        train_IVs , test_IVs, train_DV, test_DV = train_test_split(IVs, DV, test_size = 0.25, random_state = 42)
        self.classifier.fit(train_IVs, train_DV)
        
        # Use the forest's predict method on the test data
#         print(type(test_IVs))
        predictions = self.classifier.predict(test_IVs)
 
        # Train and Test Accuracy
        print('\n')
        print("Train Accuracy :: ", accuracy_score(train_DV, self.classifier.predict(train_IVs)))
        print("Test Accuracy  :: ", accuracy_score(test_DV, predictions))
        
        
        
        # Calculate the absolute errors
        errors = abs(predictions - test_DV)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))
        print('Root Mean Square Error:',round(np.sqrt(np.mean((errors)**2)),2))
        print('\n')
        print(self.classifier.best_params_)
        print(self.classifier.best_score_)