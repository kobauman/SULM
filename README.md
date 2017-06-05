# SULM -- Sentiment Utility Logistic Model

Sentiment Utility Logistic Model (SULM) is a Python library for learning users' and items' aspect profiles based on the sentiments and the overall review ratings. The full description of the model can be found in the following paper:

	Konstantin Bauman, Bing Liu, and Alexander Tuzhilin, "Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews," In Proceedings of KDD ’17, Halifax, NS, Canada, August 13-17, 2017.

## Documentation

### Initialization

SentimentUtilityLogisticModel(logger, ratings,num_aspects=len(aspects), num_factors=5, lambda_b = 0.01, lambda_pq = 0.01, lambda_z = 0.08, lambda_w = 0.01, gamma=0.001,iterations=30, alpha=0.00)

**ratings** - the list of data points, see the description of the format below. 
**num_aspects** - number of aspects in the dataset
**num_factors** - the desiresd number of latent factors in the model
**lambda_b, lambda_pq** - regularization parameter for profile coefficients (default=0.6)
**lambda_z, lambda_w**  - regularization parameter for regression weight (default=0.6)
**gamma** - the coefficient for the initial gradient descent step (default=1.0)
**iterations** - number of iterations for training the model (default=30)
**alpha** - the relative importance between rating and sentiment estimation parts (default=0.5)
**l1** - L1 normalization (default=False)
**l2** - L2 normalization (default=True)
**mult** - multiplication of general-user-item coefficients

### Input format

SULM model trains based on the ratings and sentiments that users provide for the items. 
Each entry should have the following format: [userID, itemID, overall rating, sentiment1, sentiment2, ...].
For example, if we have three aspects *['food','service','decor']* in our system and userX liked the overall experience with itemY (overall rating = 1), found *'food'* to be exceptional (sentiment1 = 1), *'service'* to be bad (sentiment2 = 0) and did not specified any opinion about *'decor'* of itemY (sentiment3 = nan), then this entry should be stored in the following format: *[userX, itemY, 1, 1, 0, nan]*.

Here is the example of the list of ratings that you can use to train the SULM model:

    ratings = [['user1','item1',1,1,1,0],
               ['user1','item2',1,0,1,0],
               ['user2','item1',0,1,0,1],
               ['user2','item2',0,np.nan,0,1],
               ['user2','item3',1,1,0,1],
               ['user3','item1',1,np.nan,0,0],
               ['user3','item2',0,np.nan,0,1],
               ['user3','item3',1,1,1,1],
               ['user4','item3',1,0,1,1],
               ['user4','item1',0,1,0,1],
               ['user4','item2',1,0,1,1]
               ]


### Functions

**average_sentiments** - Calculates average sentiments.

**sentiments_correlation** - Calculates correlation between sentiments

**train_model(self, l1 = False, l2 = True)** - Trains the model to fit the rating data set.

**predict_train** - Prints train output ONLY for testing purposes.

**predict(userID, itemID)** - Predicts ratings and sentiments for a pair of user and item (userID, itemID).
Output: rating_prediction, list of sentiment_predictions
				       
**calculate_aspect_impacts(userID, itemID, average = False, absolute = True)** - Calculates aspect impacts for a given pair of user and item (userID, itemID).
								
**predict_test(testset, filename)** - Predicts ratings and sentiments for a given set of user-item pairs and stores the result in the file.
									
**pretty_save(filename)** - Stores the model to thre file in the readable format.
							
**save(filename)** - Stores the model to the file.

**load(filename)** - Loads the model from file.

	

## Citing SULM

When [citing SULM in academic papers and theses](???), please use the following BibTeX entry:

Konstantin Bauman, Bing Liu, and Alexander Tuzhilin. 2017. Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews. In Proceedings of KDD ’17, Halifax, NS, Canada, August 13-17, 2017, 10 pages.

SULM is open source software released under the [GNU LGPLv3 license](http://www.gnu.org/licenses/lgpl.html).<br>
Copyright (c) 2017-now Konstantin Bauman