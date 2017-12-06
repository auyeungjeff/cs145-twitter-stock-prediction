import json
import nltk
import os
import math
from datetime import datetime
import matplotlib.pylab as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def test(filename):
    sid = SentimentIntensityAnalyzer()
    with open(filename, 'r') as input:
        for line in input:
            old_data = []
            old_data.append(json.loads(line))
            sentence = str(old_data[0]['text'])
            ss = sid.polarity_scores(sentence)  # Gives the sentiment score.
            scores_formatted = "{\"neg\": " + str(ss['neg']) + ", \"neu\": " + str(ss['neu']) + ", \"pos\": " + str(ss['pos']) \
                    + ", \"compound\": " + str(ss['compound']) + "}"
            print(scores_formatted)

def filter(filenameout):
    #nltk.download() #<--You may need to run this first to get it to work.
    sid = SentimentIntensityAnalyzer()

    files = os.listdir(os.getcwd())     #Looks at current directory. Change this to whatever directory your code is.
    files.sort(key=lambda x: os.path.getmtime(x))   #Sorts by time created.

    with open(filenameout, 'a') as output:
        for filename in files:
            if filename.endswith(".txt") and filename != filenameout:
                print(filename)
                with open(filename, 'r') as input:
                    for line in input:
                         try:
                            old_data = []
                            old_data.append(json.loads(line))
                            sentence = str(old_data[0]['text'])
                            ss = sid.polarity_scores(sentence)      #Gives the sentiment score.
                            scores_formatted = "{\"neg\": " + str(ss['neg']) + ", \"neu\": " + str(
                                ss['neu']) + ", \"pos\": " + str(ss['pos']) \
                                               + ", \"compound\": " + str(ss['compound']) + "}"
                            new_data = "{\"created_at\":\"" + str(
                                old_data[0]['created_at']) + "\",\"scores\":" + scores_formatted + \
                                       ",\"retweets\":" + str(old_data[0]['retweet_count'])  + "}"
                            output.write(new_data)
                            output.write("\n")
                         except:    #Code doesn't like newlines so this just passes over them if it sees one.
                            pass

def classify(filename): #get the floor
    datax = dict()
    length = dict()
    datalist = list()
    base = 0;

    with open(filename, 'r') as data:
        for tweet in data:
            old_data = []
            old_data.append(json.loads(tweet))
            timestamp1 = old_data[0]['created_at']
            t1 = datetime.strptime(timestamp1, "%a %b %d %H:%M:%S %z %Y")
            t1 = t1.replace( minute=0,second=0,microsecond=0)
            index = int(str(t1.month) + str(t1.day) + str(t1.hour))
            if base == 0:
                base = index
            if old_data[0]['scores']['compound'] != 0:
                if index in datax:
                    datax[index] = datax[index] + old_data[0]['scores']['compound']
                    length[index] += 1
                else:
                    datax[index] = old_data[0]['scores']['compound']
                    length[index] = 1

    for key in datax:
        datax[key] /= length[key]
        datax[key] = 1/(1 + math.exp(datax[key]))
        datalist.append((key,datax[key]))

    with open("logreglist.txt", 'w') as lrout:
        for item in datalist:
            print(item)
            lrout.write("{0}\n".format(item))

if __name__ == '__main__':
    filename = "logreg.txt"
    #test("code20171119-132758.txt")
    #filter(filename)
    classify(filename)
