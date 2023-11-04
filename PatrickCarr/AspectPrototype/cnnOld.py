import pandas as pan
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import sentiwordnet as sen
import csv

#REMOVE THIS!!!
#nltk.download('all')

#filler = set(stopwords.words('english'))

#stop = stopwords.words('english')

filepath = 'C:/Users/patri/cse573/AspectPrototype/test dataset.csv'
writepath = 'C:/Users/patri/cse573/AspectPrototype/filtered test dataset.csv'

file = pan.read_csv(filepath,header=0, sep=',')
#print(file.head(210))
#file['text_'] = file['text_'].str.lower().str.split()

#file['text_'] = file['text_'].apply(lambda words: [word for word in words if word not in filler])

#print(file.head(210))
print(file)

def preprocessData(file):
    reviews = word_tokenize(file.lower())
    filler = stopwords.words('english')
    filteredReviews = [word for word in reviews if word not in filler]
    lemmatizedReviews = [WordNetLemmatizer().lemmatize(word) for word in filteredReviews]
    lemmatizedReviews = [PorterStemmer().stem(word) for word in lemmatizedReviews]
    processedReviews = ' '.join(lemmatizedReviews)
    return processedReviews

def posTagging(file):
    reviews = word_tokenize(file.lower())
    taggedReviews = nltk.pos_tag(reviews)
    return taggedReviews

def setAspects(file):
    #reviews = word_tokenize(file)
    reviews = file
    index = 0
    for word in reviews:
        if word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNS' or word[1] == 'PRP' or word[1] == 'NPRP$' or word[1] == 'WP':
            reviews[index] = reviews[index] + ('a',)
        elif word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
        #elif word[1] == 'JJ':
            reviews[index] += ('s',)
            #score = sen.senti_synset(word[0]+'.a.03')
            #reviews[index] += (score.pos_score,)
            #reviews[index] += (score.neg_score,)
        elif word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
        #elif word[1] == 'VB':
            reviews[index] += ('s',)
            #score = sen.senti_synset(word[0]+'.v.03')
            #reviews[index] += (score.pos_score,)
            #reviews[index] += (score.neg_score,)
        elif word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'WRB':
        #elif word[1] == 'RB':
            reviews[index] += ('s',)
            #score = sen.senti_synset(word[0]+'.r.03')
            #reviews[index] += (score.pos_score,)
            #reviews[index] += (score.neg_score,)
        else:
            if index < len(reviews):
                reviews = reviews[:index] + reviews[index+1:]
            else:
                reviews = reviews[:index]
            index -= 1
            
        index += 1
    return reviews



file['text_'] = file['text_'].apply(preprocessData)
print('')
print(file)
file['text_'] = file['text_'].apply(posTagging) 
print('')
print(file)
file['text_'] = file['text_'].apply(setAspects) 
print('')
print(file)
#print(sen.senti_synset('breakdown.n.04'))
#print(sen.senti_synset('breakdown.n.03'))
#print(sen.senti_synset('breakdown.n.02'))
#print(sen.senti_synset('breakdown.n.01'))
#print(sen.senti_synset('breakdown'))
#print(sen.senti_synset('breakdown','n'))