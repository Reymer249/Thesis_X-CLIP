"""
    This script generates hard positives for Brittleness evaluation. It takes the original captions and generated hard
    negatives. From hard negatives it takes the 1st sentence, checks which word was changed comparing to the original
    sentence, and generates hard positive changing the same word for synonym.
"""
import json
import nltk
from nltk.corpus import wordnet as wn
import difflib

# Download necessary NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


def find_changed_word(original, modified):
    """
    Find the word that has been changed between original and modified sentences.
    Returns the changed word or None if multiple words changed.
    """
    # Tokenize sentences
    orig_tokens = original.split()
    mod_tokens = modified.split()

    # If different number of tokens, return None
    if len(orig_tokens) != len(mod_tokens):
        return None

    # Find changed tokens
    changed_tokens = []
    for orig, mod in zip(orig_tokens, mod_tokens):
        if orig.lower() != mod.lower():
            changed_tokens.append((orig, mod))

    # Return None if more than one token changed
    return changed_tokens[0] if len(changed_tokens) == 1 else None


def get_wordnet_synonyms(word, pos=None):
    """
    Get synonyms for a word using WordNet.
    If pos (part of speech) is not specified, try different POS.
    """
    synonyms = set()

    # Try different parts of speech if not specified
    pos_tags = [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV] if pos is None else [pos]

    for pos_tag in pos_tags:
        for syn in wn.synsets(word, pos=pos_tag):
            for lemma in syn.lemmas():
                # Avoid the original word and use underscores to handle multi-word synonyms
                if lemma.name().replace('_', ' ').lower() != word.lower():
                    synonyms.add(lemma.name().replace('_', ' '))

    return list(synonyms)


def generate_hard_positives(original_captions_file, hard_negatives_file):
    # Read original captions
    original_captions = {}
    with open(original_captions_file, 'r') as f:
        for line in f:
            key, caption = line.strip().split(' ', 1)
            original_captions[key] = caption

    # Read hard negatives
    with open(hard_negatives_file, 'r') as f:
        hard_negatives = json.load(f)

    # Generate hard positives
    hard_positives = {}
    for key, neg_dict in hard_negatives.items():
        # Get the original caption
        orig_caption = original_captions.get(key)
        if not orig_caption:
            continue

        # For each negative sentence
        for neg_key, neg_sentence in neg_dict.items():
            # Find the changed word
            changed_word = find_changed_word(orig_caption, neg_sentence)

            if changed_word:
                orig_word, mod_word = changed_word

                # Get synonyms
                synonyms = get_wordnet_synonyms(orig_word)

                # If synonyms found, create a positive variant
                if synonyms:
                    # Use the first synonym
                    replacement_word = synonyms[0]

                    # Replace the word in the original caption
                    words = orig_caption.split()
                    word_index = words.index(orig_word)
                    words[word_index] = replacement_word

                    # Reconstruct the sentence
                    hard_positive_sentence = ' '.join(words)

                    # Create nested dictionary structure
                    if key not in hard_positives:
                        hard_positives[key] = {}
                    hard_positives[key][neg_key] = hard_positive_sentence

    return hard_positives


# Example usage
pos = "adverb"
original_captions_file = '/vol/home/s3705609/Desktop/data_vatex/splits_txt/captions.txt'
hard_negatives_file = f'/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_chen_provided/brittleness/filtered_vatex1k5_{pos}_RE20_brit.json'
hard_positives_output_file = f'/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_chen_provided/brittleness/filtered_vatex1k5_pos_{pos}_RE20_brit.json'

# Read input files (you'll need to modify these paths)
with open(original_captions_file, 'r') as f:
    captions_content = f.read()
with open(hard_negatives_file, 'r') as f:
    negatives_content = f.read()

# Generate hard positives
hard_positives = generate_hard_positives(original_captions_file, hard_negatives_file)

# Write hard positives to output file
with open(hard_positives_output_file, 'w') as f:
    json.dump(hard_positives, f, indent=2)

print("Hard positives generated successfully!")