"""
    This module provides functionality to extract and categorize words from video captions
    based on their parts of speech (POS). It processes JSON files containing video captions,
    performs POS tagging, validates words, and organizes them into grammatical categories.
"""
import json
import nltk
import re
from tqdm import tqdm
from nltk.corpus import wordnet
from collections import defaultdict

# Download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')  # Added for word verification
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def is_valid_word(word):
    # Check if the word contains only alphabetic characters
    if not bool(re.match(r'^[a-zA-Z]+$', word)):
        return False

    # Check if it's a real English word using WordNet
    if not wordnet.synsets(word):
        return False

    return True


def extract_words_by_pos(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize dictionary to store words by part of speech
    pos_dict = {
        "noun": set(),
        "adjective": set(),
        "verb": set(),
        "adverb": set(),
        "preposition": set()
    }

    # Process each caption in the JSON data
    for video_id, captions in tqdm(data.items()):
        for caption in captions:
            # Tokenize the caption
            tokens = nltk.word_tokenize(caption.lower())

            # Tag each word with its part of speech
            tagged_words = nltk.pos_tag(tokens)

            # Categorize words based on their POS tags
            for word, tag in tagged_words:
                # Only add the word if it's valid
                if is_valid_word(word):
                    if tag.startswith('NN'):  # Nouns
                        pos_dict["noun"].add(word)
                    elif tag.startswith('JJ'):  # Adjectives
                        pos_dict["adjective"].add(word)
                    elif tag.startswith('VB'):  # Verbs
                        pos_dict["verb"].add(word)
                    elif tag.startswith('RB'):  # Adverbs
                        pos_dict["adverb"].add(word)
                    elif tag == 'IN':  # Prepositions
                        pos_dict["preposition"].add(word)

    # Convert sets to lists
    for pos in pos_dict:
        pos_dict[pos] = sorted(list(pos_dict[pos]))

    return pos_dict


# Example usage
file_path = '/captions_avail_formatted.json'
pos_dictionary = extract_words_by_pos(file_path)

# Save the result to a new JSON file
with open('../../captions_avail_formatted_dict.json', 'w') as f:
    json.dump(pos_dictionary, f, indent=4)

print("Words by part of speech have been extracted and saved to 'captions_avail_formatted_dict.json'")