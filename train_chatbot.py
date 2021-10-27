#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries and Load the Data

#import tensorflow
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import json
import pickle

intents_file = open(r'C:\Users\ADMIN\Downloads\intents.json').read()
intents = json.loads(intents_file)


# In[2]:


# Preprocessing the Data
'''
The model cannot take the raw data. It has to go through a lot of pre-processing for the machine to easily understand.
For textual data, there are many preprocessing techniques available. 
The first technique is tokenizing, in which we break the sentences into words.

By observing the intents file, we can see that each tag contains a list of patterns and responses. 
We tokenize each pattern and add the words in a list. 
Also, we create a list of classes and documents to add all the intents associated with patterns.
'''

import nltk

words = []
classes = []
documents = []

ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intents['intents']:
        #tokenize each word
        word = nltk.word_tokenize(str(pattern))
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
print(documents)


# In[3]:


'''
Another technique is Lemmatization. We can convert words into the lemma form so that we can reduce all the canonical words. 
For example, the words play, playing, plays, played, etc. will all be replaced with play. 
This way, we can reduce the number of total words in our vocabulary. 
So now we lemmatize each word and remove the duplicate words.
'''

# lemmaztize and lower each word and remove duplicates

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print(len(documents), 'documents')

# classes = intents
print(len(classes), 'classes', classes)

# words = all words, vocabulary
print(len(words), 'unique lemmatized words', words)

pickle.dump(words, open('words.pkl', 'wb'))    #To save the python object in a file, we used the pickle.dump() method.
pickle.dump(classes, open('classes.pkl', 'wb'))  #To save the python object in a file, we used the pickle.dump() method.


# In[4]:


'''
In the end, the words contain the vocabulary of our project and classes contain the total entities to classify. 
To save the python object in a file, we used the pickle.dump() method. 
These files will be helpful after the training is done and we predict the chats.
'''


# In[5]:


#Step 3. Create Training and Testing Data
'''
To train the model, we will convert each input pattern into numbers. 
First, we will lemmatize each word of the pattern and create a list of zeroes of the same length as the total number of words. 
We will set value 1 to only those indexes that contain the word in the patterns. 
In the same way, we will create the output by setting 1 to the class input the pattern belongs to.
'''

# create the training data
training = []

# create empty array for the output
output_empty = [0] * len(classes)

# training set, bag of words for every sentence
for doc in documents:
    # initializing bag of words
    bag = []

    # list of tokenized words for the pattern
    word_patterns = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.upper()) for word in word_patterns]
    # create the bag of words array with 1, if word is found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
    
# shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# create training and testing lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print('Training data is created')


# In[8]:


#Step 4. Training the Model
'''
The architecture of our model will be a neural network consisting of 3 dense layers. 
The first layer has 128 neurons, the second one has 64 and the last layer will have the same neurons as the number of classes.
The dropout layers are introduced to reduce overfitting of the model. 
We have used the SGD optimizer and fit the data to start the training of the model. 
After the training of 200 epochs is completed, we then save the trained model 
using the Keras model.save(“chatbot_model.h5”) function.
'''

# deep neural networds model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Training and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('model is created')


# In[ ]:




