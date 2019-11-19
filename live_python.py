import re
import nltk
import nltk.classify
import pickle
import customserializer
import json
import marshal
# import cPickle as pickle
# import cPickle
from pymongo import MongoClient
import csv

stopWords = []
class cleanTweets():

    def cleanTweet(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', tweet)
        tweet = re.sub('@[^\s]+', 'user', tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = tweet.strip('\'"')

        return tweet

class FeatureGen():
    def replaceTwoOrMore(self, s):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)

    def getStopWordList(self, stopWordListFileName):

        stopWords = []
        stopWords.append('user')
        stopWords.append('url')
        stopWords.append('rt')

        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
        return stopWords

    def getFeatureVector(self, tweet):
        stopWords = self.getStopWordList('stopwords.txt')
        featureVector = []
        words = tweet.split()
        for w in words:
            w = self.replaceTwoOrMore(w)
            w = w.strip('\[-.?!,":;()|0-9]')

            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            if(w in stopWords or val is None):
                continue
            else:
                featureVector.append(w.lower())
        return featureVector





class TrainingClassiffier():
    featureVector = []
    tweets = []
    def identifyFeatures(self, tweet):
        # getvec = FeatureGen()
        # featureVector = getvec.getFeatureVector(tweet)
        # featureVector = g
        # lel = 'congrats photos'
        words = set(tweet)
        features = {}
        for w in self.featureVector:
            features['contains(%s)' % w] = (w in words)
        return features

    def saveFeatureList(self, featureVector):
        saveFeatureList = open("features.pickle", "wb")
        pickle.dump(self.featureVector, saveFeatureList)
        saveFeatureList.close()

    def saveClassifier(self, classifier):
        save_classifier = open("classifier_14.csv", "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

    # def sav(self, classifier):
    #     save_classifier = open("lele.marshal", 'wb')
    #     marshal.dump(classifier, save_classifier)
    #     save_classifier.close()

    def trainClassifier (self, live):
        tweets = []
        reader = csv.reader(open('trainingData_1.csv', 'rt'))
        for row in reader:
            sentiment = row[0]
            tweet = row[1].decode('utf8')
            clean = cleanTweets()
            cleanedTweet = clean.cleanTweet(tweet)
            featureman = FeatureGen()
            features = featureman.getFeatureVector(cleanedTweet)
            self.featureVector.extend(features)
            tweets.append((features, sentiment))
        # print tweets

        self.featureVector = list(set(self.featureVector))
        hg = TrainingClassiffier()
        gh = hg.identifyFeatures
        traa = nltk.classify.util.apply_features(gh, tweets)

        # classifier = nltk.NaiveBayesClassifier.train(traa)
        self.saveFeatureList(self.featureVector)
        # self.saveClassifier(classifier)

        # save_classifier = open('classifier_20.pickle', 'wb')
        # pickle.dump(classifier, save_classifier)
        # save_classifier.close()

        classifierFile = open("classifier_21.pickle", 'rb')
        classifier = pickle.load(classifierFile)
        classifierFile.close()
        # return classifierr

        clean = cleanTweets()
        cleanedTweet = clean.cleanTweet(live)
        featureman = FeatureGen()
        test_features = featureman.getFeatureVector(cleanedTweet)
        az = TrainingClassiffier()
        qer = az.identifyFeatures(test_features)
        return classifier.classify(qer)



class Classification():

    def loadFeatureList(self):
        featureFile = open("features.pickle", "rb")
        features = pickle.load(featureFile)
        featureFile.close()
        return features

    def loadClassifier(self):
        classifierFile = open("classifier_14.csv", 'rb')
        classifierr = pickle.load(classifierFile)
        # print classifier
        classifierFile.close()
        return classifierr

    def classify(self):
        featureVector = self.loadFeatureList()
        classifierr = self.loadClassifier()

        connection = MongoClient('localhost', 27017)
        db = connection.USelection
        data = db.Hillary_10_22
        tweets_iterator = db.Hillary_10_22.find()
        pos = 0
        neg = 0
        for i in tweets_iterator:
            tweet = i['text'].encode('utf8')
            clean = cleanTweets()
            cleanedTweet = clean.cleanTweet(tweet)
            featureman = FeatureGen()
            test_features = featureman.getFeatureVector(cleanedTweet)
            az = TrainingClassiffier()
            cp = az.identifyFeatures(test_features)

            sentiment = classifierr.classify(cp)
            if sentiment=='4':
                pos = pos +1
            elif sentiment=='0':
                neg = neg + 1
            print "testTweet = %s, sentiment = %s\n" % (tweet, sentiment)
        print ("Positive:", pos)
        print ("Negative:", neg)
        print classifierr.show_most_informative_features(100)
