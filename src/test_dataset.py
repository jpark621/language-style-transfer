import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

pickle_path = '../models/ohe_char.pickle'
with open(pickle_path, 'rb') as f:
    pickle_dict = pickle.load(f)
    X1 = pickle_dict['X1']
    X2 = pickle_dict['X2']
    classes = pickle_dict['classes']

    enc = LabelEncoder()
    enc.classes_ = classes

    x1_batch = X1[0]
    x2_batch = X2[0]

    if np.argmin(x1_batch) < 0:
        x1_batch = x1_batch[:np.argmin(x1_batch)]
    if np.argmin(x2_batch) < 0:
        x2_batch = x2_batch[:np.argmin(x2_batch)]

    print(enc.inverse_transform(x1_batch))
    print(enc.inverse_transform(x2_batch))
