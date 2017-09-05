This repository helps in doing Sentiment Analysis and Topic Modelling. 

Basically, there are 4 parts to this:
1) Getting Twitter Data based upon hashtags
2) Training and saving the models (Word2Vec, TF-IDF and SVM model)
3) Using the model for Sentiment CLassification
4) Using individual sentiments to do Topic Modelling

The training data for Sentiment Classified tweets can be obtained from the below link and keep it under the folder train_data:
http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip

There are a few activities which needs to be done once:
1) under train_models - execute train_word2vec.py to train the model and save it in pickle format
2) under train_models - execute train_tfidf.py to train the model and save it in pickle format
3) under train_models - execute train_classifier.py to train the model and save it in pickle format

The Classifier accuracy is around 78% in test dataset.

Once, the above is completed, the models are ready to predict.

Keep running the twitter_data.py in order to collect more samples of data.

Once, everything is done, run all_together.py to classify the tweets into positive and negative sentiments and do a topic modelling on each dataset separately.

# Further Steps:
1) Improve the classifier by using negation statements
2) Improve the classifier using n-gram phrases
3) Use Convolution Neural Network for increased accuracy of the model
