import pandas as pd
import re
import nltk
from nltk.translate import AlignedSent, IBMModel1
# Load and Preprocess the Data
# df = pd.read_csv("./cebcen.csv")
df = pd.read_csv('engfil.tsv', sep='\t', header=None)
df.columns = ['id', 'english', 'filipino_id', 'filipino']

english_sentences = df['english'].tolist()
filipino_sentences = df['filipino'].tolist()
def clean_sentences(sentences):
    print("Cleaning sentences...")
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-zA-Z0-9]+", " ", sentence)
        cleaned_sentences.append(sentence.strip())
    return cleaned_sentences
cleaned_cebuano_sentences = clean_sentences(english_sentences)
cleaned_senama_sentences = clean_sentences(filipino_sentences)
# Train the Translation Model
def train_translation_model(source_sentences, target_sentences):
    print("Training translation model...")
    aligned_sentences = [AlignedSent(source.split(), target.split()) for source, target in zip(source_sentences, target_sentences)]
    ibm_model = IBMModel1(aligned_sentences, 10)
    return ibm_model
translation_model = train_translation_model(cleaned_cebuano_sentences, cleaned_senama_sentences)
# Translate Input Sentences
def translate_input(ibm_model):
    while True:
        source_text = input("Enter the English sentence to translate (or 'q' to quit): ")
        if source_text.lower() == 'q':
            print("Quitting...")
            break
        cleaned_text = clean_sentences(source_text.split())
        source_words = cleaned_text
        translated_words = []
        for source_word in source_words:
            max_prob = 0.0
            translated_word = None
            for target_word in ibm_model.translation_table[source_word]:
                prob = ibm_model.translation_table[source_word][target_word]
                if prob > max_prob:
                    max_prob = prob
                    translated_word = target_word
            if translated_word is not None:
                translated_words.append(translated_word)
        translated_text = ' '.join(translated_words)
        print("Translated text:", translated_text)
        print()
translate_input(translation_model)