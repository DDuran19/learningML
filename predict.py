import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('model/english_tagalog_translator_finished_1726825515.0977268.keras')

# Load tokenizers
with open('tokenizer_eng.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)
with open('tokenizer_fil.pkl', 'rb') as f:
    tokenizer_fil = pickle.load(f)

# Function to translate a sentence from English to Tagalog
def translate(sentence):
    max_seq_length = 20
    # Tokenize and pad the input sentence
    sentence_seq = tokenizer_eng.texts_to_sequences([sentence])
    sentence_seq = pad_sequences(sentence_seq, maxlen=max_seq_length, padding='post')
    
    # Predict using the model
    decoder_input = sentence_seq  # Assuming decoder input starts the same
    predicted_seq = model.predict([sentence_seq, decoder_input])
    
    # Convert predicted tokens back to words
    predicted_tokens = tf.argmax(predicted_seq[0], axis=-1).numpy()
    predicted_sentence = ' '.join([tokenizer_fil.index_word[token] for token in predicted_tokens if token > 0])
    
    return predicted_sentence

# sentence = "Where is the market?"
# translation = translate(sentence)
# print(f'Translation: {translation}')

while True:
    sentence = input('Enter a sentence in English: ')
    translation = translate(sentence)
    print(f'Translation: {translation}')