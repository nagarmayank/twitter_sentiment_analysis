from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
import gensim
import pickle

from ingest import ingest
from labelizeTweets import labelizeTweets
from tweet_tokenize import tokenize

# Get the file into a DF for training
file = r"C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\train_data\Sentiment-Analysis-Dataset\SentimentAnalysisDataset.csv"
df = ingest(file)
print("File received and processed into dataframe")

df['tokens'] = df['SentimentText'].map(tokenize)
print("Dataframe tokenization completed")

# Split the DF into training and testing
x_train, x_test, y_train, y_test = train_test_split(np.array(df.tokens), np.array(df.Sentiment), test_size=0.2)
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')
print("Dataframe split into training and test completed")

tfidf = joblib.load(r'C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\train_models\tfidf_20170821.pkl')
tweet_w2v = gensim.models.word2vec.Word2Vec.load(r'C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\train_models\w2v_model_20170821')

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


n_dim = 200

print("Build training word vectors - start")
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_train[0:800000])])
train_vecs_w2v = scale(train_vecs_w2v)
print("Build training word vectors - end")

print("Build testing word vectors - start")
test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_test[0:200000])])
test_vecs_w2v = scale(test_vecs_w2v)
print("Build testing word vectors - end")

print("SVM training started")
classifier = LinearSVC()
classifier.fit(train_vecs_w2v, y_train[0:800000])
print("SVM training complete")

y_test_pred = classifier.predict(test_vecs_w2v)
print(classifier.score(test_vecs_w2v, y_test[0:200000]))

with open('linearsvm_20170822.pkl','wb') as f:
    pickle.dump(classifier, f)