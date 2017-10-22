import sys
import getopt
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

""" Parser functions that parses and pickles the data """

def read_all_files(dir_path):
    docs = ""
    for filename in os.listdir(dir_path):
        with open(dir_path + filename, 'r') as file:
            docs += file.read()
    return docs

def one_hot_encode(dataset1, dataset2, max_length=30, char=False):
    # Train encoder on both datasets
    combined_dataset = (dataset1 + " " + dataset2).replace(".","").replace("\n", " ").split(" ")
    if char:
        combined_dataset = list((dataset1 + " " + dataset2).replace(".","").replace("\n"," "))
    enc = LabelEncoder()
    enc.fit(combined_dataset)

    # Transform both datasets
    sentences1 = dataset1.replace('.',"").split('\n')
    sentences2 = dataset2.replace('.',"").split('\n')
    
    if not char:
        sentences1 = [s.split(" ") for s in sentences1]
        sentences2 = [s.split(" ") for s in sentences2]

    X1 = matrify_sentences(sentences1, enc, max_length=max_length)
    X2 = matrify_sentences(sentences2, enc, max_length=max_length)
    classes = enc.classes_

    print(X1)

    return X1, X2, classes
    
def matrify_sentences(sentences, encoder, max_length=200):
    X = []
    for sentence in sentences:
        if not sentence:
            continue
        letters = list(sentence)
        try:
            letters = letters[:max_length]
        except:
            pass
        letters_idx = np.array(encoder.transform(letters))
        
        arr = np.full(max_length, -1)
        arr[:len(letters)] = letters_idx
        X.append(arr)
    
    return np.array(X)

def pickle_to_path(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def main(argv):
    #try:
    opts, args = getopt.getopt(argv, "hc", ["help", "char"])
    source_path1 = args[0]
    source_path2 = args[1]
    pickle_path = args[2]

    dataset1 = read_all_files(source_path1)
    dataset2 = read_all_files(source_path2)

    opts = dict(opts)
    X1, X2, classes = None, None, None
    if '-c' in opts:
        X1, X2, classes = one_hot_encode(dataset1, dataset2, char=True)
    else:
        X1, X2, classes = one_hot_encode(dataset1, dataset2)

    pickle_dict = {"X1": X1, "X2": X2, "classes": classes}
    
    pickle_to_path(pickle_dict, pickle_path)
    #except:
     #   print("Invalid arguments/flags")

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
