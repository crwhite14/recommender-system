
import math
import numpy as np
import copy
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

class RecommenderSystems:

    #remove test data from the training set
    def data_split(self, activity, user_item_indices, test_size):
    
        #create a test set by randomly removing one item from some of the users
        num_users = len(activity)
        activity_train = copy.deepcopy(activity)
        test_users = np.random.choice(num_users, test_size, replace=False)
        test_indices = np.zeros([test_size],dtype=int)

        for i,user in enumerate(test_users):
            test_indices[i] = np.random.choice(user_item_indices[user])
            activity_train[user][test_indices[i]] = 0
            
        return activity_train, test_users, test_indices
        
    #generate the predictions. can set mode = 'knn_user' or 'svd'
    def run_recommender_system(self, users, items, activity, ks, train, mode='knn_user'):
        
        num_users, num_items = activity.shape      
        test_size = num_users // len(ks)
        
        #create a list of item indices for each user
        user_item_indices = [[i for i in range(num_items) if activity[user][i] == 1]
            for user in range(num_users)]
        
        #to tune the parameter k, run the algo on each value and report accuracy
        for k in ks:
        
            #if we're training, split the data into train and test sets
            if train:
                activity_train, test_users, test_indices = self.data_split(activity, user_item_indices, test_size)
            else:
                activity_train = activity
                test_users = range(num_users)
              
            #run k-nearest neighbors or svd to generate predictions
            if mode == 'knn_user':               
                predictions = self.knn_user(activity_train, test_users, users, items, k)
            elif mode == 'svd':
                predictions = self.svd(activity_train, test_users, users, items, k)
            
            #evaluate the predictions
            if train:            
                train_error = 0
                for num, user in enumerate(test_users):
                    prob_error = 1       
                    
                    #check if our prediction matches the test-set
                    if test_indices[num] == predictions[num]:
                        prob_error = 0

                    train_error += prob_error

                prob_percent = train_error / len(test_users)
                print('k {0} error {1}'.format(k, prob_percent))
                                                               
        return predictions
            
        
    #create predictions with k-nearest neighbor
    def knn_user(self, activity_train, test_users, users, items, k):
    
        num_users, num_items = len(activity_train), len(activity_train[0])

        #create a list of items for each user, for convenience
        item_indices_train = [[i for i in range(num_items) if activity_train[user][i] == 1]
            for user in range(num_users)]      
            
        #set up k nearest neighbors    
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(activity_train) 
        predictions = []
        probs_of_predictions = []            
        
        #make predictions for each user in the test set
        for num,user in enumerate(test_users):
            
            #generate distances and indices of k closest users
            [dists], [neighbors] = knn.kneighbors(activity_train[user].reshape(1,-1), n_neighbors = k)
            dists, neighbors = dists[1:], neighbors[1:]
                           
            #add all neighbors' items to our predictions array
            item_predictions = []
            item_scores = []
            for i, neighbor in enumerate(neighbors):
                for item in item_indices_train[neighbor]:
                    if item not in item_indices_train[user]:
                        index = -1
                        if item in item_predictions:
                            index = item_predictions.index(item)
                        else:
                            item_predictions.append(item)
                            item_scores.append(0)
                            
                        #item score decays harmonically wrt distance to user
                        score =  1 / (i+1)
                        
                        #score gets a bonus if user and neighbor have same features
                        for feature in range(len(users[user])):
                            if users[user][feature] == users[neighbor][feature]:
                                score = score * 2
                        
                        item_scores[index] += score                           
            
            #compute average rating for items viewed by user
            avg_rating = 0
            for item in item_indices_train[user]:
                avg_rating += sum(items[item])
            if len(item_indices_train[user]) > 0:
                avg_rating = avg_rating / len(item_indices_train[user])
            
            #score gets a bonus if rating is close to user's average rating
            for i,item in enumerate(item_predictions):
                item_scores[i] = item_scores[i] * (1 - (sum(items[item]) - avg_rating)/11)
    
            #make a final prediction based on the highest score
            if len(item_scores) > 0:
                predictions.append(item_predictions[np.argmax(item_scores)])
            else:
                predictions.append([0])                
                 
        return predictions
 

 
    #create predictions with singular value decomposition  
    def svd(self, activity_train, test_users, users, items, k):
        
        num_users, num_items = activity_train.shape   
        
        #create a list of items for each user, for convenience        
        item_indices_train = [[i for i in range(num_items) if activity_train[user][i] == 1]
            for user in range(num_users)]
        
        #run SVD
        u, sigma, v = svds(activity_train, k=k)
        sigma = np.diag(sigma)
        predicted_activity = np.dot(np.dot(u, sigma), v)
        
        #make predictions for each user in the test set
        predictions = []
        for num,user in enumerate(test_users):        
        
            items_viewed = len(item_indices_train[user])
            top_number = min(items_viewed+1, num_items-1)
            
            #get the items with the max svd score
            top_indices = np.argpartition(predicted_activity[user], -(top_number))[-(top_number):]
            top_values = [predicted_activity[user][i] for i in top_indices]
            
            #get the first item that's not already in the user's list of items viewed
            index = items_viewed
            while top_indices[index] in item_indices_train[user]:
               index -= 1
               
            predictions.append(top_indices[index])
            
        return predictions
