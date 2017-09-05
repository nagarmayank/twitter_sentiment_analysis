import numpy as np
import joblib
import gensim

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