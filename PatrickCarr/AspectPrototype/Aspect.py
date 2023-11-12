import pandas as pan
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import sentiwordnet as sen
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product
from string import punctuation
from os import listdir
from collections import Counter
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Embedding, Dropout, LSTM, Bidirectional, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
import gensim.downloader as api
from scipy import sparse
from sklearn.utils import compute_class_weight, class_weight
from keras.callbacks import EarlyStopping
from keras import regularizers
import csv
#glove = api.load("glove-wiki-gigaword-100")

#Reference [9]ch

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=104,test_size=0.25, shuffle=True)

# filepath2 = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadataTest2.csv'
# file2 = pan.read_csv(filepath2,header=0, sep=',',encoding="utf8")

def preprocessData(file):
    reviews = word_tokenize(file.lower())
    #reviews = file
    filler = stopwords.words('english')
    filteredReviews = [word for word in reviews if word.isalpha()]
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
    pos1 = 0.25
    pos2 = 0.5
    neg1 = -0.25
    neg2 = -0.5

    #reviews = word_tokenize(file)
    reviews = file
    index = 0
    for word in reviews:
        #if word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNS' or word[1] == 'PRP' or word[1] == 'NPRP$' or word[1] == 'WP':
        if word[1] == 'NN':
            reviews[index] = reviews[index] + ('a',)
            try:
                sen.senti_synset(str(word[0])+'.n.01')
                test = True
            except:
                test = False
            if test:
                score = sen.senti_synset(str(word[0])+'.n.01')
                reviews[index] += (score.pos_score(),score.neg_score(),score.obj_score(),)
                ##print("word found")
            else:
                reviews[index] += (0,0,0,)
        #elif word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
        elif word[1] == 'JJ':
            reviews[index] += ('s',)
            try:
                sen.senti_synset(str(word[0])+'.a.01')
                test = True
            except:
                test = False
            if test:
                score = sen.senti_synset(str(word[0])+'.a.01')
                reviews[index] += (score.pos_score(),score.neg_score(),score.obj_score(),)
                ##print(word[0] + str(score.neg_score()))
                ##print("word found")
            else:
                reviews[index] += (0,0,0,)
        #elif word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
        elif word[1] == 'VB':
            reviews[index] += ('s',)
            try:
                sen.senti_synset(str(word[0])+'.v.01')
                test = True
            except:
                test = False
            if test:
                score = sen.senti_synset(str(word[0])+'.v.01')
                reviews[index] += (score.pos_score(),score.neg_score(),score.obj_score(),)
                ##print("word found")
            else:
                reviews[index] += (0,0,0,)
        #elif word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'WRB':
        elif word[1] == 'RB':
            reviews[index] += ('s',)
            try:
                sen.senti_synset(str(word[0])+'.r.01')
                test = True
            except:
                test = False
            if test:
                score = sen.senti_synset(str(word[0])+'.r.01')
                reviews[index] += (score.pos_score(),score.neg_score(),score.obj_score(),)
                ##print("word found")
            else:
                reviews[index] += (0,0,0,)
                # if ratings[index] == 1 :
                #     reviews[index] += neg2
                # elif ratings[index] == 2 :
                #     reviews[index] += neg1
                # elif ratings[index] == 3 :
                #     reviews[index] += 0
                # elif ratings[index] == 4 :
                #     reviews[index] += neg1
                # elif ratings[index] == 2 :
                #     reviews[index] += neg1
                
        else:
            if index < len(reviews):
                reviews = reviews[:index] + reviews[index+1:]
            else:
                reviews = reviews[:index]
            index -= 1
            
        index += 1
    return reviews

#vector def vectorize(file):
#     reviews = file
#     reviewData = [TaggedDocument(words=reviews,tags=[str(i)]) for i, doc in enumerate(reviews)]
#     vecModel = Doc2Vec(vector_size=20, min_count=2, epochs=40)
#     vecModel.build_vocab(reviewData)
#     vecModel.train(reviewData, total_examples=vecModel.corpus_count,epochs=vecModel.epochs)
#     #print(reviews[0])
#     #vectors = [vecModel.infer_vector(str(word)) for word in reviewData]
#     #vectors = [vecModel.infer_vector(str(reviewData[0]))]
#     #vectors = [vecModel.infer_vector(reviews)]
#     vectors = [vecModel.infer_vector(str(word)) for word in reviews]
#     for i, doc in enumerate(reviewData):
#         #print("Document", i+1, ":", doc)
#         #print("Vector:", vectors[i])
#         #print()
#     return vectors

#Reference [6]
def countVectorize(file):
    cv = CountVectorizer()
    cv.fit(file)
    vectors = cv.transform(file)
    return vectors

#Reference [5]
def tfVectorize(file):
    tv = TfidfVectorizer()
    tVectors = tv.fit_transform(file).todense()
    tVectors = tv.transform(file).todense()
    #tVectors.reshape(-1,100)
    return tVectors

#References [3] and [4]
def makeEmbedding(path, index, dim):
    vocSize = len(index)+1

    embedding = np.zeros((vocSize,dim))

    with open(path,encoding="utf8") as file:
        for line in file:
            word, *vector = line.split()
            if word in index:
                idx = index[word]
                embedding[idx] = np.array(vector, dtype=np.float32)[:dim]
    embedding
    return embedding

#Reference [11]
def makeEmbeddingAlt(path):
    embedding = {}
    with open(path,encoding="utf8") as file:
        for line in file:
            lineSplit = line.split()
            word = lineSplit[0]
            vectors = np.asarray(lineSplit[1:], dtype='float32')
            embedding[word] = vectors
    return embedding

#Reference [11]
def makeEmbeddingMatrix(embedding, dim, wordIndex):
    embeddingMatrix = np.zeros((len(wordIndex)+1,dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingAlt.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector
    return embeddingMatrix

preprocess = True
if preprocess:
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadata100.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadataTest.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadataBal.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadataBal100.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadataBal1.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/fake reviews dataset2.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadata.csv'
    filepath = 'C:/Users/patri/cse573/AspectPrototype/reviewContent+metadata10k.csv'
    file = pan.read_csv(filepath,header=0, sep=',',encoding="utf8",dtype={'text_': 'string'})
    file['text_']=file['text_'].astype(str)

    file['text_'] = file['text_'].apply(preprocessData)
    file.to_csv('C:/Users/patri/cse573/AspectPrototype/preprocessedData10k.csv', encoding='utf-8', index=False)
    print("Done preprocessing")
else:
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/preprocessedDataFull.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/preprocessedData10k.csv'
    #filepath = 'C:/Users/patri/cse573/AspectPrototype/preprocessedData.csv'
    filepath = 'C:/Users/patri/cse573/AspectPrototype/preprocessedDataBal20k.csv'
    file = pan.read_csv(filepath,header=0, sep=',',encoding="utf8",dtype={'text_': 'string','label':'float32'})
    file['text_']=file['text_'].astype(str)
    print("Done not preprocessing")

X_train, X_test, y_train, y_test = train_test_split( file['text_'],file['label'],random_state=10,test_size=0.2, shuffle=True) #test_size=0.2

#X_train = X_train.apply(preprocessData)
#X_test = X_test.apply(preprocessData)

length = 1551
tokenizer = Tokenizer(oov_token=True) #lower=False,oov_token=True
tokenizer.fit_on_texts(X_train)
print("Done tokenizer fitting")
sequence = tokenizer.texts_to_sequences(X_train)
sequenceTest = tokenizer.texts_to_sequences(X_test)
print("Done sequencing")
# print("Sequence")
# print(sequence.shape)
# print(sequenceTest.shape)
data = pad_sequences(sequence, maxlen=length)
dataTest = pad_sequences(sequenceTest, maxlen=length)
print("Done padding")
#data = pad_sequences(sequence)
wordIndex = tokenizer.word_index
#print(data)

#Reference [12]
# scalar = MinMaxScaler(feature_range=(0,1))
# scalar.fit(data)
# print("Done scale fitting")
# data = scalar.transform(data)
# dataTest = scalar.transform(dataTest)
# print("Done scale transforming")

scaler = StandardScaler()
scaler.fit(data)
print("Done scale fitting")
data = scaler.transform(data)
dataTest = scaler.transform(dataTest)
print("Done scale transforming")

labels = y_train

labels2 = y_test

#inputShape = (len(text),None,tVectors.shape[1]) #len(text[0])
#inputShape = (len(data),None,data.shape[1])
inputShape = (len(data),data.shape[0],data.shape[1])
#print(data.shape)
#print(tVectors.shape[1])

y1 = np.asarray(labels).astype('float32').reshape((-1,1))
# print(y.shape)
# print(tVectors.shape)

#dim = 100
#glovePath = 'C:/Users/patri/cse573/AspectPrototype/glove.6B.100d.txt'
dim = 300
glovePath = 'C:/Users/patri/cse573/AspectPrototype/glove.6B.300d.txt'
#embedding = makeEmbedding(glovePath, tokenizer.word_index,dim)
embeddingAlt = makeEmbeddingAlt(glovePath)
print("Done embeddingAlt")
embeddingMatrix = makeEmbeddingMatrix(embeddingAlt,dim,wordIndex)
print("Done embeddingMatrix")
#file['text_'] = file['text_'].apply(posTagging) 

#file['text_'] = file['text_'].apply(setAspects)

if dataTest.size < data.size:
    #print("test")
    dataTest = np.resize(dataTest,data.shape)
    print("Done resizing")
    #print(tVectors2.shape)


y2 = np.asarray(labels2).astype('float32')
y2 = np.resize(y2,y1.shape)

classWeight = class_weight.compute_class_weight(class_weight = "balanced", classes=np.unique(np.ravel(y1,order='C')),y=np.ravel(y1,order='C')) 
#classWeight = class_weight.compute_class_weight(class_weight = "balanced", classes=np.unique(y1),y=y1) 
#print(classWeight)
classWeight = dict(enumerate(classWeight))
#print(classWeight)
print("Done classWeight")

#kernInit = 'glorot_normal'orthogonal
kernInit = 'orthogonal'

drop = 0.2
epsil=0.001
momen=0.99
weigh=None
#Reference [7]
model = Sequential()
#model.add(Embedding(embedding.shape[0],100,weights=[embedding],input_length=tVectors.shape[1])) #,weights=[embedding]
model.add(Embedding(embeddingMatrix.shape[0],dim,weights=[embeddingMatrix],input_length=data.shape[1])) #len(wordIndex)+1
model.add(Conv1D(filters=32, kernel_size=3, input_shape=inputShape[1:],strides=1,kernel_initializer=kernInit,kernel_regularizer=regularizers.l2(0.01))) #kernel_size=3 #, padding="valid" ,kernel_initializer='glorot_normal'
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization(epsilon=epsil, momentum=momen))

model.add(Conv1D(filters=64, kernel_size=3, strides=1,kernel_initializer=kernInit,kernel_regularizer=regularizers.l2(0.01)))#, padding="valid"
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization(epsilon=epsil, momentum=momen))

model.add(Conv1D(filters=128, kernel_size=3,strides=1,kernel_initializer=kernInit,kernel_regularizer=regularizers.l2(0.01))) #, padding="valid"
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization(epsilon=epsil, momentum=momen))

#Reference [10]
#model.add(Bidirectional(LSTM(units = 50, dropout=0.25, recurrent_dropout=0.25, input_shape=inputShape[1:],return_sequences=True))) #dropout=0.25, recurrent_dropout=0.25,
#model.add(Bidirectional(LSTM(units = 50, dropout=0.25, recurrent_dropout=0.25,return_sequences=True)))
#model.add(Bidirectional(LSTM(units = 50, dropout=0.25, recurrent_dropout=0.25)))
drop2 = 0.5
model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(drop2))
#model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid')) 
#model.add(Activation('sigmoid'))#softmax sigmoid tanh

lr = 0.001
mo = 0.9
ne = True
#opt = keras.optimizers.AdamW(learning_rate=0.0001)
opt = keras.optimizers.Adam(learning_rate=0.0001)
#opt = keras.optimizers.SGD(learning_rate=lr, momentum=mo, nesterov=ne)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy']) #,'val_accuracy'

reduceLR=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=3, min_lr=0.000001, verbose=1, cooldown=0)
#earlyStopping = EarlyStopping(patience=10)
model.fit(data,y1,epochs=10, validation_split=0.25,class_weight=classWeight,shuffle = True, callbacks=[reduceLR], batch_size=16) #epochs=40  ,validation_data=(X_val, y_val)
#model.fit(data,y1,epochs=40, validation_split=0.25,class_weight=classWeight,shuffle = True, batch_size=32)
#loss, acc = model.evaluate(tVectors2,y2)
model.summary()

results1 = model.predict(dataTest)
print(results1)
results=np.where(results1 > 0.5,1,0)
print(results)

accuracy = accuracy_score(y2,results)
print("Accuracy=%f" % accuracy)
precision = precision_score(y2,results,average='macro',zero_division=0)
print("Precision=%f" % precision)
recall = recall_score(y2,results,average='macro',zero_division=0)
print("Recall=%f" % recall)
f1Score = f1_score(y2,results,average='macro',zero_division=0)
print("F1-Score=%f" % f1Score)