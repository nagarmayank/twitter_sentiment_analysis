import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from gensim.models.doc2vec import LabeledSentence

from ingest import ingest
from tweet_tokenize import tokenize

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = "%s_%s" %(label_type, i)
        labelized.append(LabeledSentence(bigram[v], [label]))
    return labelized

# Get the file into a DF for training
file = r"C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\train_data\Sentiment-Analysis-Dataset\SentimentAnalysisDataset.csv"
df = ingest(file)
print("File received and processed into dataframe")
 
df['tokens'] = df['SentimentText'].map(tokenize)
print("Dataframe tokenization completed")

phrases = Phrases(np.array(df.tokens))
bigram = Phraser(phrases)

# Split the DF into training and testing
x_train, x_test, y_train, y_test = train_test_split(np.array(df.tokens), np.array(df.Sentiment), test_size=0.2)
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')
print("Dataframe split into training and test completed")

# Train the Word2Vec model and save the model file for future use
# Set values for various parameters
corpus = [x.words for x in x_train]
num_features = 200    # Word vector dimensionality                      
min_word_count = 20   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print("Model training started")

model = Word2Vec(corpus, sg=1, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

print("Model training completed")
model_name = 'w2v_model_20170821'
model.save(model_name)
print("Word2Vec model saved")
