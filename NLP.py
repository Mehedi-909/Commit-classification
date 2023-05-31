import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('ds.csv',delimiter="#")


#drop NA values from commit text
commit_text= df['comment'].dropna()

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct



#import string

dir(string)

#give call to remove_punctations()
commit_text= commit_text.apply(lambda x: remove_punctuation(x))

commit_text.head()

#dispaly commit_text dataframe
#print(commit_text)

#tokenize

#instantiate tokenizer

#split up by spaces

tokenizer = RegexpTokenizer(r'\w+')

#call to tokenizer
commit_text = commit_text.apply(lambda x: tokenizer.tokenize(x.lower()))

#display top 100 commit messages
#print(commit_text.head(100))

def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words

#remove stop words from english
commit_text = commit_text.apply(lambda x : remove_stopwords(x))

#print starting intial commits
#print(commit_text.head(100))

#Lemmatization
lemmatizer = WordNetLemmatizer()

#        this function will is for Lemmatizing,
#        on the other hand, maps common words into one base.
# 	input	commit messages in data frame
# 	output	commit message with shorten words back to their root form
#

def word_lemmatizer(text):
    for_text= [lemmatizer.lemmatize(i) for i in text]

    return for_text

#call to lammetization
commit_text.apply(lambda x :word_lemmatizer(x))
#commit_text.to_string(index = False)
print(commit_text.head(100))
#commit_text.to_string(index = False)
#commit_text.to_string(index = False).to_csv('parsed.csv')


