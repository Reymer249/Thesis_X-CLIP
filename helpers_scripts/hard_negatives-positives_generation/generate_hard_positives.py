"""
    Generate Hard Positives file using POS methodology
"""
import json
import random
import argparse
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard positives by replacing words with synonyms.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard positives to generate per caption')
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


def get_synonyms(word, pos):
    """Get synonyms of a word"""
    synonyms = []
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():  # Exclude the original word
                synonyms.append(synonym)

    return list(set(synonyms))  # Remove duplicates


def get_hypernym_hyponym_synonyms(word, pos):
    """Get synonyms from hypernyms or hyponyms"""
    candidates = []

    # Get synsets for the word
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return []

    for syn in synsets:
        # Get synonyms from hypernyms
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    candidates.append(synonym)

        # Get synonyms from hyponyms
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    candidates.append(synonym)

    return list(set(candidates))  # Remove duplicates


def find_replaceable_words(caption):
    """Find words in caption that match any target POS"""
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    replaceable_words = {
        'NOUN': [],
        'VERB': [],
        'ADJ': [],
        'ADV': [],
        'ADP': []
    }

    # Map NLTK's POS tags to our categories
    pos_map = {
        'NOUN': ['NN', 'NNS', 'NNP', 'NNPS'],
        'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'ADJ': ['JJ', 'JJR', 'JJS'],
        'ADV': ['RB', 'RBR', 'RBS'],
        'ADP': ['IN']
    }

    for i, (word, tag) in enumerate(pos_tags):
        for pos_category, tags in pos_map.items():
            if tag in tags:
                replaceable_words[pos_category].append((i, word, tag))

    return replaceable_words, tokens


def make_replacement(replacement, word, tokens, idx):
    """Create a new sentence with the replacement word"""
    # Preserve capitalization
    new_replacement = replacement[0].upper() + replacement[1:] if word[0].isupper() else copy.deepcopy(replacement)

    # Replace the word
    new_tokens = copy.deepcopy(tokens)
    new_tokens[idx] = new_replacement

    # Reconstruct the sentence
    hard_positive = ' '.join(new_tokens)

    # Fix spacing around punctuation
    hard_positive = re.sub(r'\s+([.,!?:;])', r'\1', hard_positive)

    return hard_positive


def collect_possible_changes(replaceable_words, tokens, num_sentences):
    """Collect all possible word replacements for a caption"""
    possible_changes = []

    # First try direct synonyms for all replaceable words
    for pos_category, words in replaceable_words.items():
        for idx, word, tag in words:
            wn_pos = get_wordnet_pos(tag)
            if wn_pos:
                synonyms = get_synonyms(word.lower(), wn_pos)
                for synonym in synonyms:
                    possible_changes.append({
                        'idx': idx,
                        'word': word,
                        'replacement': synonym,
                        'source': 'direct_synonym',
                        'pos': pos_category
                    })

    # Then try hypernym/hyponym synonyms if needed
    if len(possible_changes) < num_sentences:
        for pos_category, words in replaceable_words.items():
            for idx, word, tag in words:
                wn_pos = get_wordnet_pos(tag)
                if wn_pos:
                    hyper_hypo_synonyms = get_hypernym_hyponym_synonyms(word.lower(), wn_pos)
                    for synonym in hyper_hypo_synonyms:
                        possible_changes.append({
                            'idx': idx,
                            'word': word,
                            'replacement': synonym,
                            'source': 'hyper_hypo_synonym',
                            'pos': pos_category
                        })

    return possible_changes


def generate_hard_positives(caption, num_sentences):
    """Generate hard positives for a caption"""
    replaceable_words, tokens = find_replaceable_words(caption)

    # Check if there are any replaceable words
    has_replaceable = any(len(words) > 0 for _, words in replaceable_words.items())
    if not has_replaceable:
        return []

    # Collect all possible changes
    possible_changes = collect_possible_changes(replaceable_words, tokens, num_sentences)

    # Deduplicate changes by creating a dictionary using (idx, replacement) as keys
    deduped_changes = {}
    for change in possible_changes:
        key = (change['idx'], change['replacement'])
        # Prioritize direct synonyms over hyper/hypo
        source_priority = {'direct_synonym': 0, 'hyper_hypo_synonym': 1}
        if key not in deduped_changes or source_priority[change['source']] < source_priority[
            deduped_changes[key]['source']]:
            deduped_changes[key] = change

    # Convert back to list
    possible_changes = list(deduped_changes.values())

    # Randomly select changes if we have more than needed
    if len(possible_changes) > num_sentences:
        possible_changes = possible_changes[:num_sentences]

    # Generate the sentences
    hard_positives = []
    generated_set = set()

    for change in possible_changes:
        hard_positive = make_replacement(change['replacement'], change['word'], tokens, change['idx'])

        # Skip duplicates
        if hard_positive in generated_set or hard_positive == caption:
            continue
        else:
            hard_positives.append([hard_positive, change["pos"]])
            generated_set.add(hard_positive)

    return hard_positives


def plot_distribution(distribution, num_sentences):
    """Plot distribution of number of generated sentences"""
    plt.figure(figsize=(10, 6))
    x = np.arange(num_sentences + 1)  # 0 to num_sentences
    plt.bar(x, distribution)
    plt.xlabel('Number of Generated Hard Positives')
    plt.ylabel('Count')
    plt.title('Distribution of Generated Hard Positives per Caption')
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('hard_positives_distribution.png')
    plt.close()


def main():
    args = parse_arguments()

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    hard_positives = {}

    # Initialize distribution array
    distribution = np.zeros(args.num_sentences + 1, dtype=int)

    # Process each video and its captions
    for video_id, captions in tqdm(captions_data.items()):
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"

            # Generate hard positives for this caption
            positives = generate_hard_positives(caption, args.num_sentences)

            # Update distribution count
            count = min(len(positives), args.num_sentences)
            distribution[count] += 1

            # Only add if we have any positives
            if positives:
                hard_positives[key] = positives

    # Save distribution to pickle file
    with open('hard_positives_distribution.pkl', 'wb') as f:
        pickle.dump(distribution, f)

    # Plot distribution
    plot_distribution(distribution, args.num_sentences)

    # Write output
    output_file = f"hard_positives_all_pos_{args.num_sentences}.json"
    with open(output_file, 'w') as f:
        json.dump(hard_positives, f, indent=4)

    print(f"Generated hard positives saved to {output_file}")
    print(f"Distribution saved to hard_positives_distribution.pkl and hard_positives_distribution.png")
    print(f"Distribution: {distribution}")


if __name__ == "__main__":
    main()