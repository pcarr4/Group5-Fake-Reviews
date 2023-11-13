# Installing required libraries

import pandas as pd
import numpy as np
!pip install scikit-learn gensim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import gensim

# Import the 'drive' module from the 'google.colab' library and Mount Google Drive at the specified path
from google.colab import drive
drive.mount('/content/drive')

#Loading the dataset into a variable df
df = pd.read_csv("/content/drive/MyDrive/SWM project/reviewContent+metadataBalanced.csv", encoding="ISO-8859-1",on_bad_lines='skip')
df=df.astype(str)

# Step 1: Term Frequency (TF)
#Term Frequency (TF) represents how often a word appears in a document.
tf_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
tf_matrix = tf_vectorizer.fit_transform(df['text_'])

# Step 2: Word2Vec
#Word2Vec is a technique for learning vector representations of words from large corpora.
import nltk
nltk.download('punkt')
tokenized_text = [word_tokenize(review) for review in df['text_']]
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=300, window=5, min_count=2, epochs=5)

#Creating a single feature vector referred to as word2vec_matrix, that represents the entire review using Word2Vec embeddings
#Condense the information from individual word vectors into a single representation for each review.
#This mean vector can be used as a feature in machine learning models for tasks like sentiment analysis and classification. The mean vector captures the overall semantic meaning of the words in the review.
word_vectors = []
for review in tokenized_text:
    vector = [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * 300 for word in review]
    word_vectors.append(vector)

word2vec_matrix = pd.DataFrame(word_vectors).mean(axis=0).values.reshape(1, -1)

#The Latent Dirichlet Allocation (LDA) fucntion for topic modeling
def lda_topic_distribution(reviews, num_topics, num_words_per_topic):
    # Filter out documents without meaningful words
    reviews = [review for review in reviews if len(word_tokenize(review)) > 0]

    # Create a CountVectorizer to convert the reviews into a document-term matrix
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2, max_features=5000)
    X = vectorizer.fit_transform(reviews)

    # Check if the vocabulary is not empty
    if X.shape[1] == 0:
        raise ValueError("Empty vocabulary; documents may only contain stop words.")

    # Extract feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Convert the sparse matrix to a gensim corpus
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

    # Create a mapping of word indices to words for gensim LDA model
    id2word = {idx: word for idx, word in enumerate(feature_names)}

    # Training an LDA (Latent Dirichlet Allocation) model on the corpus
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=10)

    topics = lda_model.print_topics(num_words=num_words_per_topic)
    return topics

#Creating two subsets of our DataFrame df based on the values that is either 0 or 1
fake_reviews = df[df['label'] == '0']['text_']
non_fake_reviews = df[df['label'] == '1']['text_']

#Applying the lda_topic_distribution function
fake_lda_topics = lda_topic_distribution(fake_reviews, num_topics=150, num_words_per_topic=8)
non_fake_lda_topics = lda_topic_distribution(non_fake_reviews, num_topics=200, num_words_per_topic=8)

# Merge the features into a single dataframe
features_df_fake = pd.concat([pd.DataFrame(tf_matrix.toarray()), pd.DataFrame(word2vec_matrix), pd.DataFrame(fake_lda_topics)], axis=1)
features_df_non_fake = pd.concat([pd.DataFrame(tf_matrix.toarray()), pd.DataFrame(word2vec_matrix), pd.DataFrame(non_fake_lda_topics)], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare the labels
labels_fake = np.zeros(len(fake_reviews))
labels_non_fake = np.ones(len(non_fake_reviews))

# Concatenate the features and labels
X = pd.concat([features_df_fake, features_df_non_fake], ignore_index=True)
y = np.concatenate([labels_fake, labels_non_fake])

# Concatenate the features for fake and non-fake reviews separately
X_fake = pd.concat([features_df_fake], ignore_index=True)
X_non_fake = pd.concat([features_df_non_fake], ignore_index=True)

# Concatenate the labels for fake and non-fake reviews separately
y_fake = np.zeros(len(features_df_fake))
y_non_fake = np.ones(len(features_df_non_fake))

# Concatenate the features and labels
X = pd.concat([X_fake, X_non_fake], ignore_index=True)
y = np.concatenate([y_fake, y_non_fake])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=['float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['float64']).columns

# Standardize the numeric features
numeric_transformer = StandardScaler()

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('non_num', 'passthrough', non_numeric_cols)  # Non-numeric columns are passed through without scaling
    ])

# Fit and transform on the training data
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_scaled = preprocessor.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the scaled training data
rf_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Random Forest Classifier:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate and print the classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", class_report)