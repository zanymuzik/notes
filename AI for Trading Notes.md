## NLP

### Challenges 
- Lack proper structure and semantics
- Loose rules or grammar
- Hard to parse and understand (compared to scripting and programming languages)

### Contextual Dependence
- Hard to understand for computers
- Humans have context so it is easier

### Typical NLP Pipeline
```mermaid
graph LR
A[Text Processing] --> B[Feature Extraction]; 
B-->C[Modeling]
```

### Text Processing
- Remove HTML tags
- Lowercase
- Punctuation

### Feature Extraction
- Encoding from Unicode / ASCII
- Images are easier to understand based on pixel values
- Common Tricks
	- Bag of words 
	- Word2Vec
	- Glove

### Modeling
- Designing Stastical /ML model
-  Fitting Params to Training using Optimization
-  Predict unseen data

## Text Processing
[text_processing Notebook](https://github.com/udacity/AIND-NLP/blob/master/text_processing.ipynb)
![NLP Pipeline](https://miro.medium.com/max/2000/1*ZIM9cAZY_KnJSL-T7-RTKg.png)
https://miro.medium.com/max/2000/1*ZIM9cAZY_KnJSL-T7-RTKg.png

### Capturing Text
- Text (using file IO)
- Tables (using panda)
- API (using requests library)  

### Normalization
- Capitalization (c.tolower())
- Replace punctuation with space

### Tokenization
 - Symbol that can't be split further
 - text.split() on whitespace
 - NLTK has word_tokenize - smarter than split
 - [NLTK tokenizer](http://www.nltk.org/api/nltk.tokenize.html)

### Cleaning
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to parse HTML
- Can navigate the DOM tree and extract tags from the HTML

### Stop word removal
> from nltk.corpus from stopwords 
- Reduce the vocabulary without loosing meaning

### Part of speech tagging
> from ntlk import pos_tag
- Nouns, pronouns from sentences
- converts tokens to tags

### Named Entity Recognition
> from ntlk import ne_chunk
- Identify proper nouns in sentences

### Stemming and Lemmatization
- Stemmer - porter and snowball stemmer in nltk
	- search and replace on common words
	- more efficient
- Lemmatization 
	- uses dictionary 
	- final word is meaningful (is -> be)

## Feature Extraction

### Bag of Words
- treating docs a bag of words
- convert docs to vectors
- document-term matrix containing term freq of words 
- similarity between docs = dot product of the vectors
- cosine similarity (from -1 to 1)

### TD-IDF
- not all words are equal
- frequency of terms in document as denominator 
- unique words get higher weights
> tfidf(t, d, D) = tf(t, d) . idf(t, D)
- term frequency = count (t, d) / |d|   // raw count of terms / total terms in d)
- inverse document frequency = log(|D| / |{din D : t in d}|)

### One-hot Encoding / Word Embeddings
- numerical representation for words
- similar to bag of words (with one hot, all off)
- fixed size vector for every word

### Word2vec
- Transform words to vectors
- CBoW
- Skipgram
- Robust, distributed representation
 - Fixed representation

### Glove
- Another word embedding
- P(j|i) is computed and vectors computed just that the numbers match
- Co-occurence probability values

### Embedding for deep learning
- some words can be close in multiple dimension, eg tea and coffee
- and yet, be dissimilar in another dimension
- More dimension in wordvec, make it more richer
- embeddings are way more compact than one-hot representation
- pre-trained embedding are very useful to reduce dimensions
- NLP layers are similar to pre-trained layers in CNN

###  t-SNE
- similar to PCA
- reduce dimensions to lower
- maintains relative distance between objects

## Financial Statements

### Intro


## Project 5
[NLP in stock market](https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92)
[Numpy Python Cheat Sheet.pdf](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MjczNTg2OCwzMzgyNTA5NzZdfQ==
-->