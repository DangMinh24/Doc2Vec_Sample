import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import pandas as pd

#This project is a small sample, applying doc2vec with small set of sample (1 million instances)
#Main ideas of doc2vec: convert a list of words into a specific vector,
#then compare the similarity of two sentences by compare distance between two vectors

#1/ Create a model for clustering problem
# **Note: Must determine parameter in this model depend on the user
# model=Doc2Vec(dm=1,dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2)
model=Doc2Vec(alpha=0.025,min_alpha=0.025)

#2/ Create a data set for training
# - Use all questions( question in attribute question1 and question2) in both file train and test as data for training
# - Each question is a separate "doc" need to be converted into a "vec" at some specific space

# sent_raw="What is the step by step guide to invest in share market in india?"
# # model.build_vocab(sentences=[LabeledSentence()])
# sent=LabeledSentence(words=sent_raw.split(" "),tags=[1])
# model.build_vocab(sentences=[sent])
# print(model.vocab)
# model.train(sentences=[sent])


data_train= pd.read_csv("/home/dang/Desktop/Python Project/Kaggle/Quora_Semantic_Analysis/data/train.csv")
data_test=pd.read_csv("/home/dang/Desktop/Python Project/Kaggle/Quora_Semantic_Analysis/data/test.csv")
set_questions=set()

for iter,i in enumerate(data_train["question1"]):
    set_questions.add(i)
print(len(set_questions))
for iter,i in enumerate(data_train["question2"]):
    set_questions.add(i)
print(len(set_questions))

# => See that there are some questions that are used more than once

for iter,i in enumerate(data_test["question1"]):
    if(len(set_questions)<1000000):
        set_questions.add(i)

def list_Labeled_Sentence(set_sentence):
    for iter,i in enumerate(set_sentence):
        yield LabeledSentence(words=str(i).split(" "),tags=["Sent"+str(iter)])

labeled_data=list_Labeled_Sentence(set_questions)
model.build_vocab(labeled_data)


#***Train basic first( Maintain alpha, learning rate through epochs):
# This loop try to see result of the model:
# I use an instances having tag == Sent7 (instance 8th in train data)
# "How can I be a good geologist" tag="Sent7"
# The closet vector with "Sent7" should be "Sent1007"
# "What should I do to be a great geologist?" tag="Sent1007"

for epoch in range(10):
    labeled_data=list_Labeled_Sentence(set_questions)
    model.train(labeled_data)
    print(str(epoch)+": ",end=" ")
    print(model.docvecs.most_similar("Sent7"))



#Further experiments:
# -Improve space of instance
# -Improve epochs
# -Tuning parameter (Consider how many dimensions should be appropriate)
# -How to convert cosine similarity into probability for log loss evaluation ?
# -Using workers technique to make run time faster
# -Try to decrease learning rate, alpha through each epochs

#Another approachs:
# -Feature engineering on raw question