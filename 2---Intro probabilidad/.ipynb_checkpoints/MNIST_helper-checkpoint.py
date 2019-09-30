from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
        
def plot_number(x_train, y_train, number, show_label=True, figsize=(10, 5)):
    plt.imshow(x_train[number], cmap='gray')
    if show_label:
        plt.text(0,0,str(y_train[number]), color='w', size=20, verticalalignment="top")
    plt.show()
    
def create_row(x_train, numbers):
    concatenated = x_train[numbers[0]]
    numbers=numbers[1:]
    for n in numbers:
        concatenated = np.concatenate((concatenated, x_train[n]), axis=1)
    return concatenated

def plot_numbers(x_train, numbers, columns=10, show_label=True, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    numbers = np.array(numbers).reshape(-1, columns)
    concatenated = create_row(x_train, numbers[0])
    numbers = numbers[1:,:]
    for row in numbers:
        concatenated = np.concatenate((concatenated, create_row(x_train, row)))
    plt.imshow(concatenated, cmap='gray')
    plt.show()
    
def setHistogram(data, pixels,totPixel, row):
    hist = []
    for pixel in pixels:
        prob_pixel = (data[row] == pixel).sum() / totPixel
        hist.append(prob_pixel)
    return hist
 
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')
            
def getHist(images,labels,cat,bins=256):
    priori = (labels == cat).sum()/len(labels)
    hist = np.histogram(images[labels==cat].flatten(),bins=bins,range=[0,256],density=True)[0]
    return np.log(hist),np.log(priori)

def fit(images,labels,bins=256):
    categories = set(labels)
    models = []
    for i in range (len(categories)):
        models.append(getHist(images,labels,i,bins=256))
    return models    

def predict(testImages, model):
    hists = [np.histogram(img.flatten(),bins=256,range=[0,256],density=False)[0] for img in testImages]
    predictions = [[np.matmul(m[0],hist)+m[1] for m in model] for hist in hists]
    return [np.argmax(prediction) for prediction in predictions]

def score(images,categories):
    self.predict()