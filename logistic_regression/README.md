Logistic Regression
===

Under main, there's a filename variable. That's the name of the file where all the outputs go to.
There are three functions. Run filter and classify. 
"Filter" uses a sentiment analyzer to give the tweets scores. You'll may have to do nltk.download() first run but you can comment it out later. I'm using Pycharm so it opens up a GUI and then you can download stuff. Change this line :: files = os.listdir(os.getcwd()) to whatever directory your tweets thing is in. Or if you're running the code in the same folder then you're fine. The "newdata = " line you can edit around to get the data you want. For now it stores the date created, the sentiment score of the tweet, and number of retweets.
"Classify" takes the ouput folder adds up the sentiment scores, takes an average of the compund scores, igonring 0 compound scores and throws that number into a logistic regression. 
