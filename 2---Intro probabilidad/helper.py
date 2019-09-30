import numpy as np
import math

class Model():
    def __init__(self,bins=256):
        self.bins=bins
        
    def getHist(self,images,labels,cat,bins=256):
        priori = (labels == cat).sum()/len(labels)
        hist = np.histogram(images[labels==cat].flatten(),bins=bins,range=[0,256],density=True)[0]
        return np.log(hist),np.log(priori)

    def fit(self,images,labels,bins=256):
        categories = list(set(labels))
        categories.sort()
        models = [self.getHist(images,labels,i,bins=256) for i in categories] 
        self.models = models
        self.categories = categories  

    def predict(self,testImages):
        hists = [np.histogram(img.flatten(),bins=self.bins,range=[0,256],density=False)[0] for img in testImages]
        predictions = [[np.matmul(m[0],hist)+m[1] for m in self.models] for hist in hists]
        return [np.argmax(prediction) for prediction in predictions]
        #return [self.categories[self.indexOfMax(prediction)] for prediction in predictions]

    def score(self,images,categories):
        predictions = self.predict(images)
        compare = np.equal(predictions,categories)
        nonzero = np.count_nonzero(compare)
        return nonzero/len(categories)

    def indexOfMax(self,items):
        maxv = -math.inf
        idx = -1
        for i, value in enumerate(items):
            if value > maxv:
                maxv = value
                idx =i
        return idx
