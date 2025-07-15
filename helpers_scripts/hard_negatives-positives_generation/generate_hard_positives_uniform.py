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
    parser = argparse.ArgumentParser(description='Generate hard positives by replacing words with synonyms.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--num_sentences', type=int, help='Number of hard positives to generate per caption')
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
    """Get synonyms of a word if available"""
    synonyms = []
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():  # Exclude the word itself
                synonyms.append(synonym)

    # Remove duplicates
    return list(set(synonyms))


def get_hypernym_hyponym_synonyms(word, pos):
    """Get synonyms of hypernyms or hyponyms"""
    candidates = []

    # Get synsets for the word
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return []

    for syn in synsets:
        # Get hypernyms and their synonyms
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    candidates.append(synonym)

        # Get hyponyms and their synonyms
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    candidates.append(synonym)

    return list(set(candidates))


def find_replaceable_words(caption):
    """Find words in caption that match any target POS"""
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    replaceable_words = {
        'NOUN': [],
        'VERB': [],
        'ADJ': [],
        'ADV': [],
        'PREP': []
    }

    # Map NLTK's POS tags to our categories
    pos_map = {
        'NOUN': ['NN', 'NNS', 'NNP', 'NNPS'],
        'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'ADJ': ['JJ', 'JJR', 'JJS'],
        'ADV': ['RB', 'RBR', 'RBS'],
        'PREP': ['IN']
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


def collect_all_substitutions_for_pos(pos_category, words, tokens):
    """Collect all possible substitutions for words of a given POS category"""
    direct_synonym_subs = []
    hyper_hypo_synonym_subs = []

    # 1. Collect all words with given POS
    for idx, word, tag in words:
        wn_pos = get_wordnet_pos(tag)
        if not wn_pos:
            continue

        # 2. Generate all possible direct synonym substitutions
        synonyms = get_synonyms(word.lower(), wn_pos)
        for synonym in synonyms:
            hard_positive = make_replacement(synonym, word, tokens, idx)
            direct_synonym_subs.append({
                'sentence': hard_positive,
                'source': 'direct_synonym',
                'priority': 1
            })

        # 3. Generate all possible hypernym/hyponym synonym substitutions
        hyper_hypo_synonyms = get_hypernym_hyponym_synonyms(word.lower(), wn_pos)
        for synonym in hyper_hypo_synonyms:
            hard_positive = make_replacement(synonym, word, tokens, idx)
            hyper_hypo_synonym_subs.append({
                'sentence': hard_positive,
                'source': 'hyper_hypo_synonym',
                'priority': 2
            })

    return direct_synonym_subs, hyper_hypo_synonym_subs


def generate_hard_positives_for_pos(pos_category, words, tokens, num_positives_per_pos):
    """Generate hard positives for a specific POS category"""
    if not words or num_positives_per_pos <= 0:
        return []

    # 1-3. Collect all direct synonyms and hypernym/hyponym synonyms
    direct_synonym_subs, hyper_hypo_synonym_subs = collect_all_substitutions_for_pos(
        pos_category, words, tokens
    )

    # Combine all semantic substitutions
    all_semantic_subs = direct_synonym_subs + hyper_hypo_synonym_subs

    # Remove duplicates and original sentence
    original_sentence = ' '.join(tokens)
    unique_subs = {}

    for sub in all_semantic_subs:
        sentence = sub['sentence']
        if sentence != original_sentence and sentence not in unique_subs:
            unique_subs[sentence] = sub

    # Sort by priority (direct synonyms first, then hyper/hypo)
    sorted_subs = sorted(unique_subs.values(), key=lambda x: x['priority'])

    # Take up to num_positives_per_pos substitutions
    selected_subs = sorted_subs[:num_positives_per_pos]

    # Convert to the expected format
    hard_positives = [[sub['sentence'], pos_category] for sub in selected_subs]

    return hard_positives


def generate_hard_positives(caption, num_sentences):
    """Generate hard positives for a caption using POS sampling approach"""
    replaceable_words, tokens = find_replaceable_words(caption)

    # Get available POS categories (only those with words in the sentence)
    available_pos = [pos for pos, words in replaceable_words.items() if len(words) > 0]

    if not available_pos:
        return []

    # Calculate number of positives per POS category
    num_positives_per_pos = num_sentences // 5

    # If num_sentences is not divisible by 5, we'll generate fewer sentences
    # This is acceptable as per the requirements

    all_hard_positives = []

    # Generate hard positives for each available POS category
    for pos_category in available_pos:
        pos_positives = generate_hard_positives_for_pos(
            pos_category,
            replaceable_words[pos_category],
            tokens,
            num_positives_per_pos
        )
        all_hard_positives.extend(pos_positives)

    return all_hard_positives


def main():
    args = parse_arguments()

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    hard_positives = {}

    # Process each video and its captions
    for video_id, captions in tqdm(captions_data.items()):
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"

            # Generate hard positives for this caption
            positives = generate_hard_positives(caption, args.num_sentences)

            # Only add if we have any positives
            if positives:
                hard_positives[key] = positives

    # Write output
    output_file = f"hard_positives_pos_sampled_{args.num_sentences}.json"
    with open(output_file, 'w') as f:
        json.dump(hard_positives, f, indent=4)

    print(f"Generated hard positives saved to {output_file}")


if __name__ == "__main__":
    main()