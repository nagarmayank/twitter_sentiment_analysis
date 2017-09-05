import pickle

def capture_sentiment(twitter_df, tweet_vecs):
    with open(r'C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\train_models\linearsvm_20170822.pkl', 'rb') as file:
        classifier = pickle.load(file)
    twitter_df['Sentiment'] = classifier.predict(tweet_vecs)
    return (twitter_df)