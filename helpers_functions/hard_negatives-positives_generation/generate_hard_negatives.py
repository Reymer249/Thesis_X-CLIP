import json
import random
import argparse
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard negatives by replacing words of specific POS.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--dictionary_file', help='Path to JSON file with POS dictionaries')
    parser.add_argument('--pos_to_change', choices=['noun', 'verb', 'adjective', 'adverb', 'preposition'],
                        help='Part of speech to replace')
    parser.add_argument('--num_sentences', type=int, help='Number of hard negatives to generate per caption')
    return parser.parse_args()


def get_wordnet_pos(treebank_tag):
    """Convert NLTK POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_antonym(word, pos):
    """Get antonym of a word if available"""
    antonyms = []
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([ant.name() for ant in lemma.antonyms()])

    if antonyms:
        return random.choice(antonyms).replace('_', ' ')
    return None


def get_hypernym_hyponym_antonyms(word, pos):
    """Get antonyms of hypernyms or hyponyms"""
    candidates = []

    # Get synsets for the word
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None

    for syn in synsets:
        # Get hypernyms and their antonyms
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                for ant in lemma.antonyms():
                    candidates.append(ant.name().replace('_', ' '))

        # Get hyponyms and their antonyms
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                for ant in lemma.antonyms():
                    candidates.append(ant.name().replace('_', ' '))

    if candidates:
        return random.choice(candidates)
    return None


def find_replaceable_words(caption, pos_to_change):
    """Find words in caption that match the target POS"""
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    replaceable_indices = []

    # Map the requested POS to NLTK's POS tags
    pos_map = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adjective': ['JJ', 'JJR', 'JJS'],
        'adverb': ['RB', 'RBR', 'RBS'],
        'preposition': ['IN']
    }

    target_pos_tags = pos_map.get(pos_to_change, [])

    for i, (word, tag) in enumerate(pos_tags):
        if tag in target_pos_tags:
            replaceable_indices.append((i, word, tag))

    return replaceable_indices, tokens


def generate_hard_negative(replaceable_words, tokens, pos_vocab):
    """Generate hard negative by replacing a word of the target POS"""

    # Randomly choose a word to replace
    idx, word, tag = random.choice(replaceable_words)

    # Try to get an antonym
    wn_pos = get_wordnet_pos(tag)
    replacement = None

    if wn_pos:
        # Try direct antonym
        replacement = get_antonym(word.lower(), wn_pos)

        # Try antonyms of hypernyms/hyponyms
        if not replacement:
            replacement = get_hypernym_hyponym_antonyms(word.lower(), wn_pos)

    # If no replacement found, randomly select a word of the same POS
    if not replacement and pos_vocab:
        replacement = random.choice(pos_vocab)

    if replacement:
        # Preserve capitalization
        if word[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]

        # Replace the word
        tokens[idx] = replacement

    # Reconstruct the sentence
    # Simple space joining might not be perfect for punctuation
    hard_negative = ' '.join(tokens)

    # Fix spacing around punctuation
    hard_negative = re.sub(r'\s+([.,!?:;])', r'\1', hard_negative)

    return hard_negative


def main():
    args = parse_arguments()

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Load dictionary
    with open(args.dictionary_file, 'r') as f:
        dictionary = json.load(f)

    # Get vocabulary for the target POS
    pos_vocab = dictionary.get(args.pos_to_change, [])

    hard_negatives = {}

    # Process each video and its captions
    for video_id, captions in tqdm(captions_data.items()):
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"
            hard_negatives[key] = set()
            replaceable_words, tokens = find_replaceable_words(caption, args.pos_to_change)

            if not replaceable_words:
                del hard_negatives[key]
                continue

            # Generate the requested number of hard negatives
            for j in range(args.num_sentences):
                acceptance_flag = False
                counter = 0
                while not acceptance_flag:
                    if counter > 5:
                        break
                    replaceable_words, tokens = find_replaceable_words(caption, args.pos_to_change)
                    hard_negative = generate_hard_negative(replaceable_words, tokens, pos_vocab)
                    if hard_negative not in hard_negatives[key]:
                        hard_negatives[key].add(hard_negative)
                        acceptance_flag = True
                        counter = 0
                    else:
                        counter += 1

    # Write output
    output_file = f"hard_negatives_{args.pos_to_change}.json"
    with open(output_file, 'w') as f:
        json.dump(hard_negatives, f, indent=2)

    print(f"Generated hard negatives saved to {output_file}")


if __name__ == "__main__":
    main()
