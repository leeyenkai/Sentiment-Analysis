#run code in terminal if library is not installed
#pip install azure-ai-textanalytics
#reads txt from local path

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import tokenize
import csv
from itertools import zip_longest
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#nltk.download('punkt')

key = "xxx"
endpoint = "https://xxx.cognitiveservices.azure.com/"
dfpositive = []
dfneutral = []
dfnegative = []
dfoverall = []
keywords = []

x = input("Text to analyse (0 for local path): ")
if x == str(0):
    f = open(r"[insert local path]", "r", encoding="utf8")
    temp123 = f.read()
    y = tokenize.sent_tokenize(temp123)

def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()
        
def sentiment_analysis_example(client):
    documents = [x]
    response = client.analyze_sentiment(documents=documents)[0]
    print("Document Sentiment: {}".format(response.sentiment))
    print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
        response.confidence_scores.positive,
        response.confidence_scores.neutral,
        response.confidence_scores.negative,
    ))
    for idx, sentence in enumerate(response.sentences):
        print("Sentence: {}".format(sentence.text))
        print("Sentence {} sentiment: {}".format(idx+1, sentence.sentiment))
        print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
            sentence.confidence_scores.positive,
            sentence.confidence_scores.neutral,
            sentence.confidence_scores.negative,
        ))
        dfpositive.append(sentence.confidence_scores.positive)
        dfnegative.append(sentence.confidence_scores.negative)
        dfneutral.append(sentence.confidence_scores.neutral)
        dfoverall.append(sentence.sentiment)
    global df1
    global df2
    global df3
    global df4
    df1 = dfpositive
    df2 = dfneutral
    df3 = dfnegative
    df4 = dfoverall
    return df1, df2, df3, df4

def key_phrase_extraction_example(client):
    try:
        documents = [x]
        response = client.extract_key_phrases(documents = documents)[0]
        if not response.is_error:
            print("\tKey Phrases:")
            for phrase in response.key_phrases:
                print("\t\t", phrase)
        else:
            print(response.id, response.error)
    except Exception as err:
        print("Encountered exception. {}".format(err))

def sentiment_analysis_example1(client):
    for i in y:
        documents = [i]
        response = client.analyze_sentiment(documents=documents)[0]
        print("Document Sentiment: {}".format(response.sentiment))
        print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
            response.confidence_scores.positive,
            response.confidence_scores.neutral,
            response.confidence_scores.negative,
        ))
        for idx, sentence in enumerate(response.sentences):
            print("Sentence: {}".format(sentence.text))
            print("Sentence {} sentiment: {}".format(idx+1, sentence.sentiment))
            print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
                sentence.confidence_scores.positive,
                sentence.confidence_scores.neutral,
                sentence.confidence_scores.negative,
            ))
            dfpositive.append(sentence.confidence_scores.positive)
            dfnegative.append(sentence.confidence_scores.negative)
            dfneutral.append(sentence.confidence_scores.neutral)
            dfoverall.append(sentence.sentiment)
    global df1
    global df2
    global df3
    global df4
    df1 = dfpositive
    df2 = dfneutral
    df3 = dfnegative
    df4 = dfoverall
    return df1, df2, df3, df4

def key_phrase_extraction_example1(client):
    for i in y:
        try:
            documents = [i]
            response = client.extract_key_phrases(documents = documents)[0]
            if not response.is_error:
                print("\tKey Phrases:")
                for phrase in response.key_phrases:
                    print("\t\t", phrase)
                    keywords.append(phrase)
            else:
                print(response.id, response.error)
        except Exception as err:
            print("Encountered exception. {}".format(err))
    return keywords
        


if x == str(0):
    sentiment_analysis_example1(client)
    key_phrase_extraction_example1(client)
else:       
    sentiment_analysis_example(client)
    key_phrase_extraction_example(client)

#print(df1, df2, df3)
#plt.plot(df1, label = 'Positive Sentiment')
#plt.plot(df2, label = 'Neutral Sentiment')
plt.plot(df3, label = 'Negative Sentiment')
plt.legend(loc = 'upper left')
plt.ylabel('Sentiment')
plt.show()

d = [df1,df2,df3,df4]
export_data = zip_longest(*d, fillvalue = '')

with open(r"[save path]", 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("Positive", "Neural", "Negative", "Overall"))
    wr.writerows(export_data)
myfile.close()

for n,i in enumerate(df4):
    if i == "positive":
        df4[n] = 1
    elif i == "negative":
        df4[n] = -1
    else:
        df4[n] = 0

plt.plot(df4, label = "Overall Sentiment")
plt.legend(loc='upper left')
plt.ylabel('Sentiment')
plt.show()

keywordstr = ' '.join(keywords)
wordcloud = WordCloud(background_color="white", width=1200, height=600).generate(keywordstr)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print(keywordstr)
print(keywords)

mask = np.array(Image.open(r"[save path]"))
wordcloud2 = WordCloud(background_color="grey", mode="RGBA", max_words=500, mask=mask, width=1200, height=600).generate(keywordstr)
#image_colors = ImageColorGenerator(image2)
plt.figure(figsize=[7,7])
#plt.imshow(wordcloud2.recolor(color_func=image_colors), interpolation="bilinear")
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.savefig(r"[save path]", format="png")
plt.show()
