# import packages
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm

# NLP
import spacy
# remember to do this: pip install spacy-transformers
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# sentiment with spacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob')

# visualisations and math
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    filename = os.path.join("..", "..", "..", "CDS-LANG", "tabular_examples", "fake_or_real_news.csv")
    data = pd.read_csv(filename)
    real_news = data[data["label"] == "REAL"]
    fake_news = data[data["label"] == "FAKE"]
    return real_news, fake_news

# get sentiment scores using VADER
def get_sent_scores(fake_news, real_news):
    #for fake news
    sent_fake = []
    for headline in fake_news["title"]:
        fake_scores = analyzer.polarity_scores(headline)
        sent_fake.append(fake_scores)
    # for real news
    sent_real = []
    for headline in real_news["title"]:
        real_scores = analyzer.polarity_scores(headline)
        sent_real.append(real_scores)
    # converting it to pandas dataframes
    sent_fake_df = pd.DataFrame(sent_fake)
    sent_real_df = pd.DataFrame(sent_real)
    return sent_fake_df, sent_real_df, fake_news, real_news

# get the geopolitical entities
def get_gpes(fake_news, real_news, sent_fake_df, sent_real_df):
    # start with the ones in fake news. We use nlp to find gpes
    fake_gpe = []
    
    for headline in nlp.pipe(fake_news["title"], batch_size = 500):
        for entity in headline.ents:
            if entity.label_ == "GPE":
                fake_gpe.append(entity.text)
    
    # now the same for real news
    real_gpe = []
    
    for headline in nlp.pipe(real_news["title"], batch_size = 500):
        for entity in headline.ents:
            if entity.label_ == "GPE":
                real_gpe.append(entity.text)
                
    # save the gpes to two different csv's
    # make a list
    fake_list = list(zip(fake_news["Unnamed: 0"], sent_fake_df["compound"], fake_news["title"], fake_gpe))
    # make that into a dataframe
    fake_data = pd.DataFrame(fake_list, columns = ["Text ID", "Sentiment Score", "Title", "GPE"])
    # now we do the same, but with real news
    real_list = list(zip(real_news["Unnamed: 0"], sent_real_df["compound"], real_news["title"], real_gpe))
    
    real_data = pd.DataFrame(real_list, columns = ["Text ID", "Sentiment Score", "Title", "GPE"])
    # convert this to csvs
    root = "out"
    fake_data.to_csv("out/output_fake.csv", encoding = "utf-8")
    real_data.to_csv("out/output_real.csv", encoding = "utf-8")
    return fake_data, real_data

# now we find the top 20 most common GPE's and plot them as bar charts

def top20_gpes(fake_data, real_data):
    # we begin with fake news and count each instance of each GPE
    fake_gpe_count = fake_data.value_counts("GPE")
    # then we find the top 20 most frequent
    fake_gpe_top20 = fake_gpe_count.nlargest(20)
    # make that into a list
    fake_top20 = fake_gpe_top20.tolist()
    fake_top20 = list(zip(fake_gpe_top20.index, fake_top20))
    
    # to plot, we need to define x and y
    labels, y = zip(*fake_top20)
    x = np.arange(len(labels))
    y_tics = list(range(0, 100, 10))
    # plotting the bar chart
    plt.xticks(x, labels)
    plt.yticks(y_tics)

    # trying to tidy up (this is still a bit messy)
    plt.xlabel("GPE entities")
    plt.ylabel("Frequency")
    plt.title("Top 20 GPE's in Fake News")
    plt.bar(x, y, color = "red", width = 0.8)
    plt.xticks(rotation=75)
    root = "out"
    plt.savefig("out/top20_fake_gpes.png", dpi=300, bbox_inches="tight")
    

    # now we do the same with the real news data
    real_gpe_count = real_data.value_counts("GPE")
    real_gpe_top20 = real_gpe_count.nlargest(20)
    
    real_top20 = real_gpe_top20.tolist()
    real_top20 = list(zip(real_gpe_top20.index, real_top20))
    
    # to plot, we need to define x and y
    labels, y = zip(*real_top20)
    x = np.arange(len(labels))
    y_tics = list(range(0, 100, 10))
    # plotting the bar chart
    plt.xticks(x, labels)
    plt.yticks(y_tics)

    # trying to tidy up (this is still a bit messy)
    plt.xlabel("GPE entities")
    plt.ylabel("Frequency")
    plt.title("Top 20 GPE's in Real News")
    plt.bar(x, y, color = "red", width = 0.8)
    plt.xticks(rotation=75)
    root = "out"
    plt.savefig("out/top20_real_gpes.png", dpi = 300, bbox_inches = "tight")
    return fake_top20, real_top20
                  

def main():
    real_news, fake_news = load_data()
    sent_fake_df, sent_real_df, fake_news, real_news = get_sent_scores(fake_news, real_news)
    fake_data, real_data = get_gpes(fake_news, real_news, sent_fake_df, sent_real_df)
    fake_top20, real_top20 = top20_gpes(fake_data, real_data)
                  
if __name__ == "__main__":
    main()
    


    
        

