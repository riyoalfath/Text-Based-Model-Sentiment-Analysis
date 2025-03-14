# -*- coding: utf-8 -*-

# from google.colab import drive # import dataset from gdrive on google collab
# !gdown https://drive.google.com/uc?id=1xzAV3SYVDSsR-qwrH9LIhK0GOmliwDiM # location file gdrive
# !unzip /content/archive.zip # unpack zip

import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional

def create_bigru_branch(embedding_layer, gru_units=128):
    gru_layer = GRU(gru_units)
    bigru_layer = Bidirectional(gru_layer)(embedding_layer)
    return bigru_layer

def create_model(max_sequence_length=100, vocab_size=20000, embedding_dim=200, gru_units=128,
                 num_filters=150, region_sizes=[3, 5, 7], dense_units=100, num_classes=2):
    # Input layer
    input_layer = Input(shape=(max_sequence_length,), dtype='int32')

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

    # LSTM branch
    bigru_branch = create_bigru_branch(embedding_layer, gru_units=gru_units)

    # Dense layer
    dense_layer = Dense(dense_units, activation='relu')(bigru_branch)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='AdamW', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

df_train = pd.read_csv('/content/Corona_NLP_train.csv', encoding = 'latin')
df_test = pd.read_csv('/content/Corona_NLP_test.csv', encoding = 'latin')

df_train.head()

X_train = df_train['OriginalTweet']
y_train = df_train['Sentiment']

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re

# Clean words
def text_cleaner(tweet):
    # remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    # remove non english
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    # remove html tags
    tweet = re.sub(r'<.*?>',' ', tweet)
    # remove website url
    tweet = re.sub(r'(http|https)://[=a-zA-Z0-9_/?&.-]+',' ', tweet)
    # remove digits
    tweet = re.sub(r'\d+',' ', tweet)
    # remove hashtags
    tweet = re.sub(r'#\w+',' ', tweet)
    # remove mentions
    tweet = re.sub(r'@\w+',' ', tweet)
    # lowercase
    tweet = tweet.lower()
    #removing stop words
    tweet = tweet.split()
    tweet = " ".join([word for word in tweet if not word in stop_words])
    return tweet

stop_words = stopwords.words('english')

X_train = X_train.apply(text_cleaner)
X_train.head()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X = tokenizer.texts_to_sequences(X_train)
vocab_size = len(tokenizer.word_index)+1

X = pad_sequences(X, padding='post')

sentiments = {'Extremely Negative': 0,
            'Negative': 0,
            'Neutral': 1,
            'Positive':2,
            'Extremely Positive': 2
           }
y_train = y_train.map(sentiments)
labels = ['Negative', 'Neutral', 'Positive']

y_train

# Build the model
model = create_model(max_sequence_length=X.shape[1], vocab_size=vocab_size, embedding_dim=100,
                     gru_units=128, num_filters=15, region_sizes=[3], dense_units=20,
                     num_classes=3)

model.summary()

y = to_categorical(y_train)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Extracting history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Creating epochs range
epochs = range(1, len(train_loss) + 1)

# Plotting loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss (BiGRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

X_test = df_test['OriginalTweet'].copy()
y_test = df_test['Sentiment'].copy()

X_test = X_test.apply(text_cleaner)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, padding='post', maxlen = X.shape[1])

y_test = y_test.map(sentiments)

y_test = to_categorical(y_test)

y_test.shape

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'BiGRU Test Accuracy: {accuracy * 100:.2f}%')