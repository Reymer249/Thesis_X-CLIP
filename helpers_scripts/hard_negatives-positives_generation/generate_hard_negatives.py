"""
    Generates Hard Negatives file using POS methodology
"""
import json
import random
import argparse
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
import re
import copy

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard negatives by replacing words of specific POS.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--dictionary_file', help='Path to JSON file with POS dictionaries')
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


def get_antonyms(word, pos):
    """Get antonym of a word if available"""
    antonyms = []
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([ant.name() for ant in lemma.antonyms()])

    if antonyms:
        return [ant.replace('_', ' ') for ant in antonyms]
    return []


def get_hypernym_hyponym_antonyms(word, pos):
    """Get antonyms of hypernyms or hyponyms"""
    candidates = []

    # Get synsets for the word
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return []

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

    return candidates


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
    hard_negative = ' '.join(new_tokens)

    # Fix spacing around punctuation
    hard_negative = re.sub(r'\s+([.,!?:;])', r'\1', hard_negative)

    return hard_negative


def collect_possible_changes(replaceable_words, tokens, pos_vocabs, num_sentences):
    """Collect all possible word replacements for a caption"""
    possible_changes = []

    # First try direct antonyms for all replaceable words
    for pos_category, words in replaceable_words.items():
        for idx, word, tag in words:
            wn_pos = get_wordnet_pos(tag)
            if wn_pos:
                antonyms = get_antonyms(word.lower(), wn_pos)
                for antonym in antonyms:
                    possible_changes.append({
                        'idx': idx,
                        'word': word,
                        'replacement': antonym,
                        'source': 'direct_antonym',
                        'pos': pos_category
                    })

    # Then try hypernym/hyponym antonyms if needed
    if len(possible_changes) < num_sentences:  # Arbitrary limit to avoid collecting too many
        for pos_category, words in replaceable_words.items():
            for idx, word, tag in words:
                wn_pos = get_wordnet_pos(tag)
                if wn_pos:
                    hyper_hypo_antonyms = get_hypernym_hyponym_antonyms(word.lower(), wn_pos)
                    for antonym in hyper_hypo_antonyms:
                        possible_changes.append({
                            'idx': idx,
                            'word': word,
                            'replacement': antonym,
                            'source': 'hyper_hypo_antonym',
                            'pos': pos_category
                        })

    return possible_changes


def generate_random_replacements(replaceable_words, pos_vocabs, num_needed):
    """Generate random word replacements from vocabulary"""
    random_replacements = []

    # Flatten the replaceable words across all POS categories
    all_replaceable = []
    for pos_category, words in replaceable_words.items():
        for word_info in words:
            all_replaceable.append((word_info, pos_category))

    if not all_replaceable:
        return []

    # Generate random replacements
    for _ in range(num_needed):
        (idx, word, tag), pos_category = random.choice(all_replaceable)
        replacement = random.choice(pos_vocabs[pos_category])
        random_replacements.append({
            'idx': idx,
            'word': word,
            'replacement': replacement,
            'source': 'random_vocab',
            'pos': pos_category
        })

    return random_replacements


def generate_hard_negatives(caption, num_sentences, pos_vocabs):
    """Generate hard negatives for a caption"""
    replaceable_words, tokens = find_replaceable_words(caption)

    # Check if there are any replaceable words
    has_replaceable = any(len(words) > 0 for _, words in replaceable_words.items())
    if not has_replaceable:
        return []

    # Collect all possible changes
    possible_changes = collect_possible_changes(replaceable_words, tokens, pos_vocabs, num_sentences)

    # If we still don't have enough changes, add random replacements
    if len(possible_changes) < num_sentences:
        num_needed = num_sentences - len(possible_changes)
        random_replacements = generate_random_replacements(replaceable_words, pos_vocabs, num_needed)
        possible_changes.extend(random_replacements)

    # Deduplicate changes by creating a dictionary using (idx, replacement) as keys
    deduped_changes = {}
    for change in possible_changes:
        key = (change['idx'], change['replacement'])
        # Prioritize direct antonyms over hyper/hypo, and those over random
        source_priority = {'direct_antonym': 0, 'hyper_hypo_antonym': 1, 'random_vocab': 2}
        if key not in deduped_changes or source_priority[change['source']] < source_priority[
            deduped_changes[key]['source']]:
            deduped_changes[key] = change

    # Convert back to list
    possible_changes = list(deduped_changes.values())

    # Randomly select changes if we have more than needed
    if len(possible_changes) > num_sentences:
        possible_changes = possible_changes[:num_sentences]

    # Generate the sentences
    hard_negatives = []
    generated_set = set()

    for change in possible_changes:
        hard_negative = make_replacement(change['replacement'], change['word'], tokens, change['idx'])

        # Skip duplicates
        if hard_negative in generated_set or hard_negative == caption:
            continue
        else:
            hard_negatives.append([hard_negative, change["pos"]])
            generated_set.add(hard_negative)

    return hard_negatives


def main():
    args = parse_arguments()

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Load dictionary
    with open(args.dictionary_file, 'r') as f:
        pos_vocabs = json.load(f)

    hard_negatives = {}

    # Process each video and its captions
    for video_id, captions in tqdm(captions_data.items()):
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"

            # Generate hard negatives for this caption
            negatives = generate_hard_negatives(caption, args.num_sentences, pos_vocabs)

            # Only add if we have any negatives
            if negatives:
                hard_negatives[key] = negatives

    # Write output
    output_file = f"hard_negatives_all_pos_{args.num_sentences}.json"
    with open(output_file, 'w') as f:
        json.dump(hard_negatives, f, indent=4)

    print(f"Generated hard negatives saved to {output_file}")


if __name__ == "__main__":
    main()