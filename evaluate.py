import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model/english_tagalog_translator_finished_1726825515.0977268.keras')

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

test_eng_seq = data['test_eng_seq']
test_fil_seq = data['test_fil_seq']

# Shift decoder inputs and outputs for test data
test_decoder_input = test_fil_seq[:, :-1]
test_decoder_output = test_fil_seq[:, 1:]

# Evaluate the model
loss = model.evaluate([test_eng_seq, test_decoder_input], test_decoder_output)
print(f'Test Loss: {loss}')
