#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time

#Variables that contains the user credentials to access Twitter API 
#access_token = "<insert_your_access_token>"
#access_token_secret = "<insert_your_access_token_secret>"
#consumer_key = "<insert_your_consumer_key>"
#consumer_secret = "<insert_your_consumer_secret>"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        output = open(r"C:\Users\mayank.nagar\Desktop\ML\twitter_analysis\twitter_output.txt","a")
        output.write(data)
        output.write("\n")
        output.close()
        time.sleep(30)
        return True

    def on_error(self, status):
        print("Error")
      
if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['westernunion','western union','WesternUnion','Western Union'])
