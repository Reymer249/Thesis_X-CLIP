"""
    Generates Hard Negatives file using POS methodology following the distribution of occurrence of the part-of-speech
    in the sentences

    EX: if we need to generate 20 hard negatives, we try to generate 4 for every POS tag present. So if we have only
    [VERB, NOUN, ADJ], the total number of generated hard negatives will be 4*3 = 12 < 20, but we are okay with that,
    as in the end we will receive the proper distribution of hard negatives (same as distribution of occurrence of
     the respective part-of-speech in the sentences)

    Algorithm (when already given a sentence and a pos tag):
        1) collect all the words with given pos
        2) generate all possible antonym substitutions
        3) if it is not enough, add antonyms of a hypernyms or hyponyms of the words
        4) if it is not enough, calculate how many substitutions we are missing => n
        5) n times substitute a random word with given pos with the word with a given pos from the dictionary

        therefore, no max_attemps variables or anything like that

        when retrieving values from the set collected on the step 5, the priority should be given first to the
        substitutions with direct antonyms, then antonyms of hypernyms or hyponyms, and then the rest. so we should
        keep track how we generate substitution ad prioritize on that
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
    hard_negative = ' '.join(new_tokens)

    # Fix spacing around punctuation
    hard_negative = re.sub(r'\s+([.,!?:;])', r'\1', hard_negative)

    return hard_negative


def collect_all_substitutions_for_pos(pos_category, words, tokens, pos_vocabs):
    """Collect all possible substitutions for words of a given POS category"""
    direct_antonym_subs = []
    hyper_hypo_antonym_subs = []

    # 1. Collect all words with given POS
    for idx, word, tag in words:
        wn_pos = get_wordnet_pos(tag)
        if not wn_pos:
            continue

        # 2. Generate all possible direct antonym substitutions
        antonyms = get_antonyms(word.lower(), wn_pos)
        for antonym in antonyms:
            hard_negative = make_replacement(antonym, word, tokens, idx)
            direct_antonym_subs.append({
                'sentence': hard_negative,
                'source': 'direct_antonym',
                'priority': 1
            })

        # 3. Generate all possible hypernym/hyponym antonym substitutions
        hyper_hypo_antonyms = get_hypernym_hyponym_antonyms(word.lower(), wn_pos)
        for antonym in hyper_hypo_antonyms:
            hard_negative = make_replacement(antonym, word, tokens, idx)
            hyper_hypo_antonym_subs.append({
                'sentence': hard_negative,
                'source': 'hyper_hypo_antonym',
                'priority': 2
            })

    return direct_antonym_subs, hyper_hypo_antonym_subs


def generate_random_substitutions(pos_category, words, tokens, pos_vocabs, num_needed):
    """Generate random substitutions from vocabulary"""
    random_subs = []

    # Generate n random substitutions
    for _ in range(num_needed):
        # Randomly select a word position and a replacement word
        idx, word, tag = random.choice(words)
        replacement = random.choice(pos_vocabs[pos_category])

        hard_negative = make_replacement(replacement, word, tokens, idx)
        random_subs.append({
            'sentence': hard_negative,
            'source': 'random_vocab',
            'priority': 3
        })

    return random_subs


def generate_hard_negatives_for_pos(pos_category, words, tokens, pos_vocabs, num_negatives_per_pos):
    """Generate hard negatives for a specific POS category"""
    if not words or num_negatives_per_pos <= 0:
        return []

    # 1-3. Collect all direct antonyms and hypernym/hyponym antonyms
    direct_antonym_subs, hyper_hypo_antonym_subs = collect_all_substitutions_for_pos(
        pos_category, words, tokens, pos_vocabs
    )

    # Combine all semantic substitutions
    all_semantic_subs = direct_antonym_subs + hyper_hypo_antonym_subs

    # 4. Check if we need random substitutions
    num_missing = max(0, num_negatives_per_pos - len(all_semantic_subs))

    # 5. Generate random substitutions if needed
    random_subs = []
    if num_missing > 0:
        random_subs = generate_random_substitutions(
            pos_category, words, tokens, pos_vocabs, num_missing
        )

    # Combine all substitutions
    all_substitutions = all_semantic_subs + random_subs

    # Remove duplicates and original sentence
    original_sentence = ' '.join(tokens)
    unique_subs = {}

    for sub in all_substitutions:
        sentence = sub['sentence']
        if sentence != original_sentence and sentence not in unique_subs:
            unique_subs[sentence] = sub

    # Sort by priority (direct antonyms first, then hyper/hypo, then random)
    sorted_subs = sorted(unique_subs.values(), key=lambda x: x['priority'])

    # Take up to num_negatives_per_pos substitutions
    selected_subs = sorted_subs[:num_negatives_per_pos]

    # Convert to the expected format
    hard_negatives = [[sub['sentence'], pos_category] for sub in selected_subs]

    return hard_negatives


def generate_hard_negatives(caption, num_sentences, pos_vocabs):
    """Generate hard negatives for a caption using POS sampling approach"""
    replaceable_words, tokens = find_replaceable_words(caption)

    # Get available POS categories (only those with words in the sentence)
    available_pos = [pos for pos, words in replaceable_words.items() if len(words) > 0]

    if not available_pos:
        return []

    # Calculate number of negatives per POS category
    num_negatives_per_pos = num_sentences // 5

    # If num_sentences is not divisible by 5, we'll generate fewer sentences
    # This is acceptable as per the requirements

    all_hard_negatives = []

    # Generate hard negatives for each available POS category
    for pos_category in available_pos:
        pos_negatives = generate_hard_negatives_for_pos(
            pos_category,
            replaceable_words[pos_category],
            tokens,
            pos_vocabs,
            num_negatives_per_pos
        )
        all_hard_negatives.extend(pos_negatives)

    return all_hard_negatives


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
    output_file = f"hard_negatives_pos_sampled_{args.num_sentences}.json"
    with open(output_file, 'w') as f:
        json.dump(hard_negatives, f, indent=4)

    print(f"Generated hard negatives saved to {output_file}")


if __name__ == "__main__":
    main()