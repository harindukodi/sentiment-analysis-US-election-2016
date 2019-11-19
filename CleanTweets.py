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
    def replaceTwoOrMore(self, s):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)
        # word = re.sub(r'([a-z])\1+', r'\1', s)
        # return word

    def getFeatureVector(self, tweet):
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
    #end

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
    st = open('stopwords.txt', 'r')
    stopWords = getStopWordList('stopwords.txt')

    def find_bigrams(self, input_list): #removed self from the arguments
        bigram_list = []
        for i in range(len(input_list)-1):
            bigram_list.append(input_list[i]+" "+input_list[i+1])
        return bigram_list

    def identifyFeatures(self, tweet):
        words = set(tweet)
        features = {}
        for w in self.featureList:
            features['contains(%s)' % w] = (w in words)
        return features

class TrainingClassiffier():

    def trainClassifier (self):
        tweets = {}
        reader = csv.reader(open('traningData.csv','rt'))
        for row in reader:
            sentiment = row[0];
            tweet = row[1].decode('utf8')
            clean = cleanTweets()
            cleanedTweet = clean.cleanTweet(tweet)
            featureman = FeatureGen()
            features = featureman.getFeatureVector(cleanedTweet)
            # generate feature list
            self.featureList.extend(features)
            # genearate annotated tweet list for training
            tweets.append((features,sentiment))
        #training the classifier
        classifier = self.startTraining()
        # save feature list
        self.saveFeatureList(self.featureList)
        #save Classifier
        self.saveClassifier(self.classifier)

    def startTraining(self):
        idenFeatures = FeatureGen()
        mark_feature = idenFeatures.identifyFeatures()
        self.featureList = list(set(self.featureList))
        training_data = nltk.classify.util.apply_features(mark_feature, self.tweets)
        return nltk.NaiveBayesClassifier.train(training_data)

    def saveFeatureList (self):
        saveFeatureList = open("features.pickle","wb");
        pickle.dump(self.featureList, saveFeatureList)
        saveFeatureList.close()

    def saveClassifier (self, classifier):
        save_classifier = open("classifier.pickle","wb");
        pickle.dump(classifier, save_classifier)
        save_classifier.close()


class Classification():

    def loadFeatureList(self):
        # specify the file
        featureFile = open("features.pickle","rb")
        # retrieve feature list from the file
        features = pickle.load(featureFile)
        featureFile.close()
        return features

    def loadClassifier (self):
        # specify the file
        classifierFile = open("classifier.pickle","rb")
        # retrive classifier from file
        classifier = pickle.load(classifierFile)
        classifierFile.close()
        return classifier

    def classify (self, tweet):
        featureList = self.loadFeatureList()
        classifier = self.loadClassifier()
        clean = cleanTweets()
        cleanedTweet = clean.cleanTweet(tweet)
        featureManage = FeatureGen()
        # take feature list
        features = featureManage.generateFeatureList(cleanedTweet)
        # identify features
        markedFeatures = featureManage.identifyFeatures (features)
        return classifier.classify(markedFeatures)
