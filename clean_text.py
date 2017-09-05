from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer

def clean(doc):
    lm = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    stop.add('western')
    stop.add('union')
    stop.add('westernunion')
    stop.add('username')
    stop.add('hashtag')
    stop.add('url')
    stop.add('wu')
    stop.add('emoji')
    exclude = set(string.punctuation)
    #english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    
    clean_text = re.sub(r'http\S+', '$URL$ ', doc)
    clean_text = re.sub(r'\$\w*','',clean_text)
    clean_text = re.sub(r'['+string.punctuation+']+', ' ',clean_text)
    clean_text = re.sub(r'@\w*','$USERNAME$ ',clean_text)
    clean_text = re.sub(r'#\w*','$HASHTAG$ ',clean_text)
    clean_text = re.sub(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])','$EMOJI$ ',clean_text)
    clean_text = re.sub(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])','$EMOJI$ ',clean_text)
    clean_text = ''.join(ch for ch in clean_text if ch not in exclude)
    clean_text = " ".join([i for i in clean_text.lower().split() if i not in stop and len(i) > 2])
    normalized = " ".join(lm.lemmatize(word) for word in clean_text.split())
    #normalized = " ".join([i for i in clean_text.lower().split() if i not in stop and len(i) > 2 and i in english_vocab])
    return normalized