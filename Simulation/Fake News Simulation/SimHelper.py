#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

def generate_weighted_percentage(percentage):
    """
    return either 1 which is fake or 0 which is real given a certain percentage
    """
    try:
        return bernoulli.rvs(percentage)
    except Exception as e:
        print(str(e))
        
def get_beta_probs(a,b):
    size = 10000
    s4 = []
    s3 = []
    s2 = []
    s1 = []
    for _ in range(1,100):
        data_expon = beta.rvs(a,b,size=size)
        x = np.interp(data_expon, (data_expon.min(), data_expon.max()), (1, 5))
        s4.append(len(np.where(x>=4)[0]))
        s3.append(len(np.where((x>=3) & (x<4))[0]))
        s2.append(len(np.where((x>=2) & (x<3))[0]))
        s1.append(len(np.where((x>=1) & (x<2))[0]))
    
    return round(np.mean(s1)/size,2),round(np.mean(s2)/size,2),round(np.mean(s3)/size,2),round(np.mean(s4)/size,2)

def positive_or_negative(mu=0.5):
    """
    input range [0,1]
    returns either 1 or -1
    """
    return 1 if bernoulli.rvs(mu) ==1 else -1

def generate_weighted_sentiment(a=.8,b=1.4,mu=0.5):
    """
    return a number between 1 and 5 which is the absolute sentiment (1-5)
    """
    data_beta = beta.rvs(a,b,size=1000)
    x = np.interp(data_beta, (data_beta.min(), data_beta.max()), (1, 5))
    return round(random.choice(x)* positive_or_negative(mu),2)

def generate_num_of_shares(loc,scale):
    """
    return a number of shares given a mean and standard deviation
    """
    num_shares = expon.rvs(loc=loc,scale=scale)
    return int(num_shares)

def generate_bias(mu,sigma):
    """
    return a bias given a mean and standard deviation
    """
    sign = positive_or_negative(abs(mu))
    mu = mu*10
    bias = random.gauss(mu,sigma) * sign
    #print(bias)
    bias = bias/10
    if bias<-1:
        return -1
    elif bias>1:
        return 1
    return round(bias,2)

def generate_preference(mu=0,sigma=0.3,upper=1,lower=-1):
    z = np.random.normal(mu,sigma)
    while z > upper or z < lower:
        z = np.random.normal(mu,sigma)
    return z

def generate_preference_polarized(mu,sigma):
    pos = generate_preference(mu=mu,sigma=sigma)
    neg = generate_preference(mu=mu,sigma=sigma)*-1
    if abs(pos) == max(abs(pos),abs(neg)):
        return pos
    return neg

def average_degree_of_graph(G):
    try:
        return float(nx.info(G).split(':')[-1].strip())
    except:
        print('Could not find graph, returned 0')
        return 0
    
def nodes_connected(G,node1, node2):
    return node1 in G.neighbors(node2)

def get_random_neighbor(G,user,original_user=None):
#     print('Neighbors of user:',user,'are:',list(G.neighbors(user)))
    neighbors= list(G.neighbors(user))
    neighbors = [ item for item in neighbors if (int(item)/1000000) >7]
    
    #this is for finding a second neighbor and not returning the same user
    if original_user:
        neighbors.remove(original_user)
    
    random_user = random.choice(neighbors)
    return random_user

def get_second_random_neighbor(G,user):
    
    extended_neighbors = set()
    
    neighbors= list(G.neighbors(user))
    shuffle(neighbors)
    
    for neighbor in neighbors:
        [extended_neighbors.add(x) for x in list(G.neighbors(neighbor)) if x not in neighbors]
        
    extended_neighbors = list(extended_neighbors)
    if user in extended_neighbors:
        extended_neighbors.remove(user)
    
    if not extended_neighbors:
        return None
    else:
        return random.choice(extended_neighbors)

def create_initial_connections(G,hist_display=True,graph_display=True):
    """
    This function gets a graph of users and then creates edges between the nodes based on an exponential function
    """
    
    #determine the size of the graph
    size = len(list(G.nodes()))
    
    smallest_percentage_leftout = randint(size*.15,size*.35)
    size = size-smallest_percentage_leftout
#     print(size,smallest_percentage_leftout)
    
    #determine the number of connections each node will have
    data_expon = expon.rvs(loc=1,scale=500000,size=size)
    x = np.interp(data_expon, (data_expon.min(), data_expon.max()), (size*0.1, size*0.6))
    x = x.astype(int)
    
    size = len(list(G.nodes()))
    for i in range(smallest_percentage_leftout):
        x = np.append(x,randint(1,int(size*.09)))
#     print(len(x))
    
    if hist_display:
        plt.hist(x,bins=10)

    neighbor_edges = {}
    counter = 0
    for node in list(G.nodes):
        neighbor_edges[node] = x[counter]
        counter += 1
    
#     print(neighbor_edges)

    """for each node, get all its existing connections, then find the number of assigned conncetions
    the difference between these two numbers is how many connections you need to make for that node
    also remove unwanted nodes you already visited"""
    counter = 0
    for node in tqdm(list(G.nodes)):
        try:
            universe = [x for x in list(G.nodes) if x not in list(G.neighbors(node)) and neighbor_edges[x]>0]
            if len(list(G.neighbors(node))) >= x[counter]:
                counter += 1
                continue

            for i in range(x[counter]):
                neighbor = random.choice(universe)
                while (neighbor == node or neighbor in list(G.neighbors(node)) or neighbor_edges[neighbor]<1):
                    universe.remove(neighbor)
#                     print('Removing',neighbor,'new length=',len(universe))
                    neighbor = random.choice(universe)
#                 print(node,neighbor)
                neighbor_edges[neighbor] -= 1
                neighbor_edges[node] -= 1
                G.add_edge(node,neighbor)

            counter += 1

        except:
#             print('no more nodes')
            pass

#     print(G.edges())
    if graph_display:
#         pos = nx.spring_layout(G, scale=2)  # double distance between all nodes
        pos = nx.spring_layout(G,k=0.2,iterations=50)
        plt.figure(2,figsize=(20,20))
        nx.draw(G, pos,node_size=300)
        plt.show()

    return G