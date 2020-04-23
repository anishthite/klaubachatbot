# from nltk.tokenize import sent_tokenize, word_tokenize
# import gensim
# from gensim.models import Word2Vec
# import json
# import pickle
# import numpy as np
# from scipy import spatial
from nltk.metrics.distance import edit_distance
# functions used for training
def read_file():
    sample = open("words.txt", "r") 
    s = sample.read() 
    # Replaces escape character with space 
    f = s.replace("\n", " ") 
    return f

# def tokenize_setup(data_list):
#     data = []
#     print data_list
#     for i in sent_tokenize(data_list): 
#         temp = [] 
#         for j in word_tokenize(i): 
#             temp.append(j.lower()) 
#         data.append(temp)
#     return data

# def train_wtov(data):
#     model = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)
#     #word_model = open()
#     model.save('word_model.model')
#     return model



# # functions used for similarity checking
# def similar(query,commandvec, model, index2word_set):
#     vec = avg_sentence_vector(query, model, 100, index2word_set)
#     #print(type(vec))
#     print commandvec
#     print 1 - spatial.distance.cosine(vec, commandvec)
#     return 1 - spatial.distance.cosine(vec, commandvec)

# def avg_sentence_vector(words, model, num_features, index2word_set):
#     #function to average all words vectors in a given paragraph
#     featureVec = np.zeros((num_features,), dtype="float32")
#     nwords = 0
#     for word in words:
#         if word in index2word_set:
#             nwords = nwords+1
#             featureVec = np.add(featureVec, model[word])
#     if nwords>0:
#         featureVec = np.divide(featureVec, nwords)
#     return featureVec


#functions used for finding best match

def read_command_file():
    mylist = []
    sample = open("commands.txt", "r") 
    s = sample.read() 
    mylist = list(s.splitlines())
    print mylist
    return mylist

# def generate_pickle_command_dic(commandlist):
#     #read vectors into memory
#     mydic = {}
#     model = gensim.models.Word2Vec.load('word_model.model')
#     index2word_set = set(model.wv.index2word)
#     for command in commandlist:
#         mydic[tuple(avg_sentence_vector(command, model, 100, index2word_set))] = command
#     with open('commanddic.pickle', 'wb') as handle:
#         pickle.dump(mydic, handle)

# def compare(query, commandlist, model, indexset):
#     numlist = range(len(commandlist))
#     #return similar(query,commandlist[index], model, indexset)
#     return max(numlist, key= lambda index: similar(query,commandlist[index], model, indexset))
    
# def load_compare(query):
#     with open('commanddic.pickle', 'rb') as handle:
#         commanddic = pickle.load(handle)
#         commandlist = [tup for tup in commanddic.keys()]
        
#         model = gensim.models.Word2Vec.load('word_model.model')
#         index2word_set = set(model.wv.index2word)
#         return compare(query, commandlist, model, index2word_set)

#TODO: train on new dic, map to command, map to reply, actual reply code 



#EDIT DISTANCE BASED MODEL, SMALLER

def distance_model(query):
    commandlist = read_command_file()
    numlist = range(len(commandlist))
    return min(numlist, key = lambda index: edit_distance(query, commandlist[index]))

def map_output(index):
    print read_command_file()[index]

if __name__ == '__main__':
    #train w2v code
    #model = train_wtov(tokenize_setup(read_file()))
    
    #generate command dic code
    # generate_pickle_command_dic(read_command_file())

    #run code
    while True:
        query = raw_input('Enter query')
        #import time
        #start = time.time()
        print map_output(distance_model(query))
        #print map_output(load_compare(query))
        #end = time.time()
        #print(end - start)
