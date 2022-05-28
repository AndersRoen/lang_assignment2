# lang_assignment2

github repo: https://github.com/AndersRoen/lang_assignment2.git

## Language analytics assignment 2 description
Assignment 2 - Sentiment and NER

We saw this week that an common application of NLP in cultural data is finding what is known as named entities. In previous sessions, we also saw some simple ways of calculating a sentiment score for a stretch of natural language. These tasks allow us to extract some potentially interesting information from our data - what exactly is being spoken about in a text and what kind of sentiment does the text show?

For this assignment, you will write a small Python program to perform NER and sentiment analysis using the techniques you saw in class. You have the choice of one of two different tasks:

    Using the corpus of English novels, write some code which does the following tasks
        For every novel in the corpus
            Get the sentiment of the first sentence.
            Get the sentiment of the final sentence.
            Plot the results over time, with one visualisation showing sentiment of opening sentences over time and one of closing sentences over time.
            Find the 20 most common geopolitical entities mentioned across the whole corpus - plot the result as a bar chart.

OR

    Using the corpus of Fake vs Real news, write some code which does the following
        Split the data into two datasets - one of Fake news and one of Real news
        For every headline
            Get the sentiment scores
            Find all mentions of geopolitical entites
            Save a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in that text
        Find the 20 most common geopolitical entities mentioned across each dataset - plot the results as a bar charts

For this assignment, you should create a private Github repository and add me as a collaborator. When submitting via Brightspace, simply send the link to the repository; I will provide feedback and comments via Github's built in functionality.

## methods
I chose to do the second option.
First, the script loads in the corpus and splits it up into two datasets, ```real``` and ```fake```. Then, for every headline in each dataset, it gets the sentiment scores, using the ```VADER``` ```polarity_scores``` function. Then, using ```NER``` it finds all of the ```GPEs``` in the headlines. Then, each headline along with its GPE and sentiment score are saved to a csv. This results in two csv files, one for ```real_news``` and one for ```fake_news```.
Then, the script finds the 20 most frequent GPEs in each dataset, plots them to a bar chart and saves the plot.

## Usage
This script is quite simple to run. First you need to put the ```real_fake_news``` corpus into the ```in``` folder. Also make sure that you have downloaded the proper spacy model with ```python -m spacy download en_core_web_trf```. Then, you need to run the ```setup_lang.sh``` script. Then, point the command line to the ```lang_assignment2``` folder and run the ```assignment2.py``` script from the ```src``` folder.

## Results
The bar charts show that most fake news in the corpus are related to Russia and the US. Due to the way the NLP pipeline does NER tagging, ```America```, ```US```, ```U.S``` is all tagged as different entities, though they should be the same. A larger model might help with that. The real news results are more related to the US and Iran, with the same type of tagging problems. Many American states are also tagged as seperate from the US, which is more acceptable. This shows us that the corpus is very centered on the us, which makes sense since presumably the corpus is made from largely American news. 
