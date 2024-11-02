import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')


df = df.drop(columns=['statement', 'BinaryNumTarget'])

# Extract the tweet text and labels
tweets = df['tweet'].astype(str).values
labels = df['majority_target'].values

# Pre-train Word2Vec model
sentences = [tweet.split() for tweet in tweets]  # Tokenize the tweets for Word2Vec
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Extract the vocabulary and word vectors
word_vectors = word2vec_model.wv
vocab_size = len(word_vectors)
embedding_dim = word_vectors.vector_size

# Create an embedding matrix
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))  # +1 for padding token

# Populate the embedding matrix with the Word2Vec vectors
for i, word in enumerate(word_vectors.index_to_key):
    embedding_matrix[i + 1] = word_vectors[word]  # i+1 because index 0 is reserved for padding

# Tokenize and convert the tweets to sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad the sequences to ensure they are of the same length
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Modify the model to add more complexity
model = Sequential([
    Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),  # Use pre-trained Word2Vec embeddings, freeze the layer
    LSTM(128, return_sequences=True),  # Increased units
    Dropout(0.5),  # Keep Dropout
    LSTM(64, return_sequences=False),  # Added another LSTM layer
    Dropout(0.5),  # Keep Dropout
    Dense(32, activation='relu'),  # L2 Regularization
    Dropout(0.5),  # Additional Dropout
    Dense(1, activation='sigmoid')  # Binary classification
])

# Implement Early Stopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Compile and fit with increased epochs
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train,
    epochs=20,  # Increased number of epochs
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]  # Use the same callbacks
)

# Train the model with validation data and callbacks
history = model.fit(
    X_train, y_train,
    epochs=10,  # Increase the number of epochs for thorough training
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]  # Add the callbacks
)

# Display the model summary
model.summary()
