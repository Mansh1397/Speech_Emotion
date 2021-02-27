import helper as hlp
import sys
import numpy as np
import joblib

file = sys.argv[1]

def pred(file):
    t=[]
    feats = hlp.extract_feature(file, mfcc=True, chroma=True, mel=True)
    t.append(feats)
    return np.array(t)
    
arr = pred(file)
model = joblib.load('Model')
print('Model loaded')
print('*************************')
print("Sentiment of Speaker:\n", model.predict(arr)[0])
