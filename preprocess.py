import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('engfil.tsv', sep='\t', header=None)
df.columns = ['id', 'english', 'filipino_id', 'filipino']

# Extract English and Filipino sentences
english_sentences = df['english'].values
filipino_sentences = df['filipino'].values

# Train-test split (80% train, 20% test)
train_eng, test_eng, train_fil, test_fil = train_test_split(english_sentences, filipino_sentences, test_size=0.2, random_state=42)

# Tokenization
max_vocab_size = 10000
max_seq_length = 20

tokenizer_eng = Tokenizer(num_words=max_vocab_size, filters='', lower=True)
tokenizer_fil = Tokenizer(num_words=max_vocab_size, filters='', lower=True)

tokenizer_eng.fit_on_texts(train_eng)
tokenizer_fil.fit_on_texts(train_fil)

train_eng_seq = tokenizer_eng.texts_to_sequences(train_eng)
train_fil_seq = tokenizer_fil.texts_to_sequences(train_fil)
test_eng_seq = tokenizer_eng.texts_to_sequences(test_eng)
test_fil_seq = tokenizer_fil.texts_to_sequences(test_fil)

# Pad sequences
train_eng_seq = pad_sequences(train_eng_seq, maxlen=max_seq_length, padding='post')
train_fil_seq = pad_sequences(train_fil_seq, maxlen=max_seq_length, padding='post')
test_eng_seq = pad_sequences(test_eng_seq, maxlen=max_seq_length, padding='post')
test_fil_seq = pad_sequences(test_fil_seq, maxlen=max_seq_length, padding='post')

# Save processed data and tokenizers
import pickle
with open('tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(tokenizer_eng, f)
with open('tokenizer_fil.pkl', 'wb') as f:
    pickle.dump(tokenizer_fil, f)

# Save preprocessed data
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump({
        'train_eng_seq': train_eng_seq,
        'train_fil_seq': train_fil_seq,
        'test_eng_seq': test_eng_seq,
        'test_fil_seq': test_fil_seq
    }, f)
