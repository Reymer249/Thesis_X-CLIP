import nltk
from nltk.corpus import wordnet


def get_synonyms(word):
    """
    Retrieve synonyms for a given word using WordNet.

    Args:
        word (str): The word to find synonyms for.

    Returns:
        list: A list of unique synonyms for the word.
    """
    # Download necessary NLTK data if not already downloaded
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    # Get all synsets (sets of synonyms) for the word
    synsets = wordnet.synsets(word)

    # Collect unique synonyms
    synonyms = set()
    for synset in synsets:
        # Get lemma names (words) for each synset
        for lemma in synset.lemmas():
            # Add the lemma name if it's different from the original word
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)

    return list(synonyms)


def main():
    # Example usage
    while True:
        user_word = input("Enter a word to find synonyms (or 'quit' to exit): ").strip()

        if user_word.lower() == 'quit':
            break

        try:
            word_synonyms = get_synonyms(user_word)

            if word_synonyms:
                print(f"\nSynonyms for '{user_word}':")
                for synonym in word_synonyms:
                    print(f"- {synonym}")
            else:
                print(f"No synonyms found for '{user_word}'.")

        except Exception as e:
            print(f"An error occurred: {e}")

        print()  # Extra newline for readability


if __name__ == "__main__":
    main()