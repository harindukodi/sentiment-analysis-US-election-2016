import json
import pymongo
import tweepy
import csv
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import live_python as s

consumer_key = "DAMeDDQNzAtEL6YRC2jMJ8t9i"
consumer_secret = "pgQKv4qkhKx7Bklau7kgnwP6sxsBvkCszWpOtIleRCmOhonwtM"
access_key = "803431624851517440-ouoS6Oo9MHK7rjRgeaG5MZO9DA9GqPK"
access_secret = "WTyc9QWTRQuUXVwi8y1YsP9tL152hgv1MhRVSezd4NmK4"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

class listener(StreamListener):
    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data["text"].encode('utf-8')
            we = s.TrainingClassiffier()
            sentiment_value = we.trainClassifier(tweet)
            print(tweet.decode('utf-8'), sentiment_value)


            with open('twitter_data.csv','wb') as myfile:
                wrtr = csv.writer(myfile, delimiter=',', quotechar='"')
                # for row in tweet:
                wrtr.writerow([tweet,sentiment_value])
                myfile.flush() # whenever you want


            output = open("twitter-out3.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
            return True
        except:
            return True

    def on_error(self, status):
        print(status)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["hillary"], languages=["en"])
