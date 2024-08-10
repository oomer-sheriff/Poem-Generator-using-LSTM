import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import cmudict
import os

def read_poems_from_folder(folder_path):
    all_poems = ""
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the contents of each file
            with open(file_path, 'r', encoding='utf-8') as file:
                poem_content = file.read()
                
            # Add a newline between poems and append to the main string
            all_poems += poem_content + "\n\n"
    
    return all_poems


folder_path = "haiku"
text = read_poems_from_folder(folder_path)

# Then proceed with your existing code, starting with tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1



# Download the CMU Pronouncing Dictionary for rhyming
nltk.download('cmudict')
pronouncing_dict = cmudict.dict()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# Save the model in .keras format
model.save('lstm_poem_model.keras')

# Function to get rhyming words
def get_rhyming_words(word, word_list):
    if word.lower() not in pronouncing_dict:
        return []
    
    pronunciations = pronouncing_dict[word.lower()]
    rhymes = []
    
    for w in word_list:
        if w.lower() in pronouncing_dict:
            w_pronunciations = pronouncing_dict[w.lower()]
            for pron1 in pronunciations:
                for pron2 in w_pronunciations:
                    if pron1[-2:] == pron2[-2:]:
                        rhymes.append(w)
                        break
                if w in rhymes:
                    break
    
    return rhymes

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        if (_ + 1) % 5 == 0:  # Every 5th word should rhyme
            last_word = seed_text.split()[-5]
            rhyming_words = get_rhyming_words(last_word, list(tokenizer.word_index.keys()))
            if rhyming_words:
                rhyming_probs = [predicted_probs[tokenizer.word_index[w]] for w in rhyming_words if w in tokenizer.word_index]
                if rhyming_probs:
                    predicted_word = rhyming_words[np.argmax(rhyming_probs)]
                else:
                    predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
            else:
                predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        else:
            predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        
        seed_text += " " + predicted_word
    
    return seed_text