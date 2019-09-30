import newspaper
import requests
import re 
from collections import Counter
from newspaper import fulltext
def body(url):
    html = requests.get(url).text
    #Get string from URl, and set every char to lower 
    textFromHtml = fulltext(html).lower()
    #Get only a-z from string
    cleanString = re.sub(r"[^a-z]+", "%", textFromHtml)
    textList = re.split(' |%',cleanString)
    dictionary = Counter(textList)
    #Print beauty format string
    for key,value in dictionary.items():
        print("La cantidad de veces que aparece la palabra {} es {}".format(key,value))
        