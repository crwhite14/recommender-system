# recommender-system
Recommender system using SVD and KNN

This repository contains a recommender system for items given a dataset of user features, item features, and user-item interactions. We use two different collaborative filtering approaches: matrix factorization and user-based nearest neighbors. The best parameters for our models are learned using cross-validation. Our final model is a hybrid approach, combining the interaction data with the user features and item features.

Our dataset consists of three files. The users dataset consists of user-id, continent, and gender. The item dataset consists of item-id and item rating. The interactions dataset consists of user-item interactions, for example, user i viewed item j while browsing items online.

In order to find the best model and model parameters, we need a way to evaluate a model's performance. Evaluating a recommender system's performance is a bit less straightforward than, e.g., evaluating a supervised binary classifier. The approach we take is as follows. We randomly choose 1/k users, and randomly remove one item from their interaction list. The 1/k pairs of users and items is the validation set. The rest of the data (all the data except for the 1/k total items removed) is the training set.

We train our model on the training set, and then for each user in the test set, we output predictions for the next item the user interacted with. Our model's validation error is the average number of times the predicted item does not match the corresponding item in the validation set. When choosing between e.g. five different parameters, we set k=5 and run the above approach.

We tried user-based nearest neighbor search and matrix factorization. These two approaches had comparable error, around 90%. Note that random guessing would have error 1/66=98%, since there are 66 items total. We used nearest-neighbor search to generate our final predictions. This algorithm works as follows. For each user i, we compute the k closest users by using the cosine distance in the activity matrix and the user features. We create a list of item predictions consisting of the items of the k closest neighbors. Each item j receives a score based on the number of neighbors who interacted with j, and the distance of those neighbors to i. We also increase the score of item j if its features are close to the average features of items that user i interacted with. Our prediction for user i is the item j with the maximum score out of the items that user i has not already interacted with.



