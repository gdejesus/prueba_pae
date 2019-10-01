import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from ipykernel import kernelapp as app
import helperGauss as helper

class GaussModel:
    def __init__(self,labels,e=1e-8):
        self.epsilon=e

    def fit(self,images,labels):
        self.categories = len(list(set(labels)))
        print(len(list(set(labels))))
        self.means = list()
        self.stds = list()
        self.models = list()
        for cat in range(self.categories):
            self.means = np.mean(images[labels==cat])
            self.stds = np.std(images[labels==cat])
            self.models.append((self.means,self.stds))

    def predict(self,testImages):
        # .pdf funcion de densidad de probabilidad
        predictions = []
        for cat in range(self.categories):
            media = self.models[cat][0]
            desvio = self.models[cat][1]+self.epsilon
            Gaussian = norm(media,desvio)
            #Solo si tenemos igual cantidad de valores sobre cada lista, podemos no castearlo a np.array
            predictions.append(np.log(Gaussian.pdf(testImages)).sum(axis=1))
        #predictions = np.array(predictions)
        return np.argmax(predictions, axis=0)
        
    def score(self,images,categories):
        predictions = self.predict(images)
        compare = np.equal(predictions,categories)
        nonzero = np.count_nonzero(compare)
        return nonzero/len(categories)