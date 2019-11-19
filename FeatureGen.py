import re
import nltk
import pickle
import cPickle
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
    # featureVector = []
    def replaceTwoOrMore(self, s):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)
        # word = re.sub(r'([a-z])\1+', r'\1', s)
        # return word

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
    # st = open('stopwords.txt', 'r')
    # stopWords = getStopWordList('stopwords.txt')

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


    def genBigrams(self, unigrams):
        bigrams = []
        for i in range(len(unigrams)-1):
            bigrams.append(unigrams[i]+" "+unigrams[i+1])
        return bigrams

    def identifyFeatures(self, tweet, featureVector):
        # getvec = FeatureGen()
        # featureVector = getvec.getFeatureVector()
        featureVector = g
        words = set(tweet)
        features = {}
        for w in featureVector:
            features['contains(%s)' % w] = (w in words)
        return features

class TrainingClassiffier():
    featureVector = []
    tweets = []
    def startTraining(self):
        idenFeatures = FeatureGen()
        mark_feature = idenFeatures.identifyFeatures()
        self.featureVector = list(set(self.featureVector))
        training_data = nltk.classify.util.apply_features(mark_feature, self.tweets)
        return nltk.NaiveBayesClassifier.train(training_data)

    def trainClassifier (self):
        tweets = []
        reader = csv.reader(open('traningData.csv','rt'))
        for row in reader:
            sentiment = row[0]
            tweet = row[1].decode('utf8')
            clean = cleanTweets()
            cleanedTweet = clean.cleanTweet(tweet)
            featureman = FeatureGen()
            features = featureman.getFeatureVector(cleanedTweet)
            self.featureVector.extend(features)
            tweets.append((features, sentiment))
        self.classifier = self.startTraining()
        self.saveFeatureList(self.featureVector)
        self.saveClassifier(self.classifier)


    def saveFeatureList(self, featureVector):
        saveFeatureList = open("features.pickle", "wb")
        pickle.dump(self.featureVector, saveFeatureList)
        saveFeatureList.close()

    def saveClassifier(self, classifier):
        save_classifier = open("classifier.pickle", "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

class Classification():

    def loadFeatureList(self):
        # specify the file
        featureFile = open("features.pickle", "rb")
        # retrieve feature list from the file
        features = pickle.load(featureFile)
        featureFile.close()
        return features

    def loadClassifier(self):
        # specify the file
        classifierFile = open("classifier.pickle", "rb")
        # retrive classifier from file
        classifier = pickle.load(classifierFile)
        classifierFile.close()
        return classifier

    def classify(self, tweet):
        featureVector = self.loadFeatureList()
        classifier = self.loadClassifier()
        clean = cleanTweets()
        cleanedTweet = clean.cleanTweet(tweet)
        featureManage = FeatureGen()
        features = featureManage.getFeatureVector(cleanedTweet)
        markedFeatures = featureManage.identifyFeatures(features)
        return classifier.classify(markedFeatures)
