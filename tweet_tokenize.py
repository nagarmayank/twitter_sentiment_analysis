from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

def tokenize(tweet):
    try:
        #tweet = unicode(tweet.decode('utf-8').lower())
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return list(tokens)
    except:
        return 'NC'