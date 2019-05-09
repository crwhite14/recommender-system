
from recommender_systems import RecommenderSystems

import math
import numpy as np
import matplotlib.pyplot as plt


#create an array of user features
def load_users():

    f = open("data/users.txt", "r")
    users = []
    num_features = 0
    for index,line in enumerate(f):
    
        #the first row is the header
        if index == 0:
            num_features = len(line.split())
        else:
            features = [int(num) for num in line.split()]
            users.append(features)
        
    return users
    
#create an array of item features
def load_items():

    f = open("data/items.txt", "r")
    items = []
    num_features = 0    
    for index,line in enumerate(f):
    
        #the first row is the header
        if index == 0:
            num_features = len(line.split())
        else:
            features = [int(num) for num in line.split()]
            items.append(features)
    
    return items

#create a matrix of user-item activity
def load_interactions(num_users, num_items):

    f = open("data/interactions.txt", "r")
    activity = np.zeros([num_users, num_items])
    
    for index,line in enumerate(f):
    
        #first row is the header
        if index == 0:
            continue

        user_id, item_id = line.split()
        user_id, item_id = int(user_id), int(item_id)
        activity[user_id-1][item_id-1] = 1 
            
    return activity
    
#output array of predictions to a text file
def output_predictions(predictions):

    f = open("data/predictions.txt", "w")
    f.write("user\titem\n")
    f.writelines(str(index+1)+'\t'+str(predictions[index]+1)
        +'\n' for index in range(len(predictions)))
    f.close()


#load all input data
users = load_users()
items = load_items()
num_users, num_items = len(users), len(items)
activity = load_interactions(num_users, num_items)

rs = RecommenderSystems()

#use cross-validation to find the best value of k for k-nearest neighbors
print('starting cross-validation for k-nearest neighbors')
ks = [100, 200, 300, 400, 500]
predictions = rs.run_recommender_system(users, items, activity, ks, train=True, mode='knn_user')

#use cross-validation to find the best value of k for singular value decomposition
print('starting cross-validation for SVD')
ks = [2, 3, 4, 5, 6]
predictions = rs.run_recommender_system(users, items, activity, ks, train=True, mode='svd')

#generate final predictions using k-nearest neighbors where k=500
print('generating predictions for k-nearest neighbors with k=300')
predictions = rs.run_recommender_system(users, items, activity, [300], train=False, mode='knn_user')

#write predictions to file
print('outputting predictions to data/predictions.txt')
output_predictions(predictions)
    
