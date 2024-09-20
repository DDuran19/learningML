import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import time

import pickle

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_eng_seq = data['train_eng_seq']
train_fil_seq = data['train_fil_seq']

# Shift decoder inputs and outputs
train_decoder_input = train_fil_seq[:, :-1]
train_decoder_output = train_fil_seq[:, 1:]

# best is 16

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


checkpoint_callback = ModelCheckpoint(
    filepath='model/english_tagalog_translator_best.keras',
    monitor='val_loss',     # Monitor validation loss
    save_best_only=True,    # Only save the model if val_loss improves
    save_weights_only=False # Save full model, not just weights
)
# Hyperparameters
embedding_dim = 256
units = 512
batch_size = 64
vocab_size_eng = 10000
vocab_size_fil = 10000
max_seq_length = 20

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size_eng, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size_fil, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_fil, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

try:
    # Train the model
    model.fit([train_eng_seq, train_decoder_input], train_decoder_output, batch_size=batch_size, epochs=20, validation_split=0.2, callbacks=[checkpoint_callback, early_stopping])
    # Save the trained model
    model.save(f'model/english_tagalog_translator_finished_{time.time()}.keras')
except KeyboardInterrupt:
    model.save(f'model/english_tagalog_translator_interrupted_{time.time()}.keras')

