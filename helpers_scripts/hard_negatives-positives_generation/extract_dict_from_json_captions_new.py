"""
    This module provides functionality to extract and categorize words from video captions
    based on their parts of speech (POS). It processes JSON files containing video captions,
    performs POS tagging, validates words, and organizes them into grammatical categories.
    Enhanced to filter processing based on specific video IDs from a text file.
"""
import json
import nltk
import re
from tqdm import tqdm
from nltk.corpus import wordnet
from collections import defaultdict
import os

# Download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')  # Added for word verification
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def load_video_ids(video_ids_file_path):
    """
    Load video IDs from a text file.

    Args:
        video_ids_file_path (str): Path to the text file containing video IDs

    Returns:
        set: Set of video IDs to process
    """
    if not os.path.exists(video_ids_file_path):
        print(f"Warning: Video IDs file '{video_ids_file_path}' not found. Processing all videos.")
        return None

    video_ids = set()
    try:
        with open(video_ids_file_path, 'r') as file:
            for line in file:
                video_id = line.strip()
                if video_id:  # Skip empty lines
                    video_ids.add(video_id)

        print(f"Loaded {len(video_ids)} video IDs from '{video_ids_file_path}'")
        return video_ids

    except Exception as e:
        print(f"Error reading video IDs file: {e}")
        print("Processing all videos instead.")
        return None


def is_valid_word(word):
    """
    Check if a word is valid (alphabetic characters only and exists in WordNet).

    Args:
        word (str): Word to validate

    Returns:
        bool: True if word is valid, False otherwise
    """
    # Check if the word contains only alphabetic characters
    if not bool(re.match(r'^[a-zA-Z]+$', word)):
        return False

    # Check if it's a real English word using WordNet
    if not wordnet.synsets(word):
        return False

    return True


def extract_words_by_pos(json_file_path, video_ids_file_path=None):
    """
    Extract words from video captions and categorize them by part of speech.

    Args:
        json_file_path (str): Path to JSON file containing video captions
        video_ids_file_path (str, optional): Path to text file containing video IDs to process

    Returns:
        dict: Dictionary containing words categorized by POS
    """
    # Load video IDs to filter (if provided)
    target_video_ids = load_video_ids(video_ids_file_path) if video_ids_file_path else None

    # Read the JSON file
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        print(f"Loaded captions data with {len(data)} videos")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

    # Initialize dictionary to store words by part of speech
    pos_dict = {
        "NOUN": set(),
        "ADJ": set(),
        "VERB": set(),
        "ADV": set(),
        "PREP": set()
    }

    # Filter data based on target video IDs if provided
    if target_video_ids:
        original_count = len(data)
        # Filter to only include videos that are in our target list
        filtered_data = {vid: captions for vid, captions in data.items() if vid in target_video_ids}
        print(f"Filtered from {original_count} to {len(filtered_data)} videos based on video IDs file")

        # Check for missing video IDs
        missing_ids = target_video_ids - set(data.keys())
        if missing_ids:
            print(f"Warning: {len(missing_ids)} video IDs from the filter file were not found in the captions data")
            if len(missing_ids) <= 10:  # Only show first 10 missing IDs to avoid clutter
                print(f"Missing IDs: {', '.join(sorted(list(missing_ids)))}")

        data = filtered_data
    else:
        print("No video ID filter provided. Processing all videos.")

    if not data:
        print("No videos to process after filtering.")
        return pos_dict

    # Process each caption in the filtered JSON data
    total_captions = sum(len(captions) for captions in data.values())
    print(f"Processing {total_captions} captions from {len(data)} videos...")

    processed_captions = 0
    for video_id, captions in tqdm(data.items(), desc="Processing videos"):
        for caption in captions:
            processed_captions += 1

            # Tokenize the caption
            tokens = nltk.word_tokenize(caption.lower())

            # Tag each word with its part of speech
            tagged_words = nltk.pos_tag(tokens)

            # Categorize words based on their POS tags
            for word, tag in tagged_words:
                # Only add the word if it's valid
                if is_valid_word(word):
                    if tag.startswith('NN'):  # NOUNs
                        pos_dict["NOUN"].add(word)
                    elif tag.startswith('JJ'):  # ADJs
                        pos_dict["ADJ"].add(word)
                    elif tag.startswith('VB'):  # VERBs
                        pos_dict["VERB"].add(word)
                    elif tag.startswith('RB'):  # ADVs
                        pos_dict["ADV"].add(word)
                    elif tag == 'IN':  # PREPs
                        pos_dict["PREP"].add(word)

    # Convert sets to lists and sort them
    for pos in pos_dict:
        pos_dict[pos] = sorted(list(pos_dict[pos]))

    # Print summary statistics
    print("\nExtraction Summary:")
    for pos, words in pos_dict.items():
        print(f"{pos}: {len(words)} unique words")

    total_unique_words = sum(len(words) for words in pos_dict.values())
    print(f"Total unique words: {total_unique_words}")

    return pos_dict


def main():
    """
    Main function to run the word extraction process.
    """
    # File paths
    base = "/vol/home/s3705609/Desktop/data_vatex/splits_txt"
    captions_file_path = os.path.join(base, 'captions_avail_train_020.json')
    video_ids_file_path = os.path.join(base, 'vatex_train_avail_020.txt')  # Path to your video IDs file
    output_file_path = os.path.join(base, 'captions_avail_train_020_dict.json')

    # Extract words by POS with video ID filtering
    pos_dictionary = extract_words_by_pos(captions_file_path, video_ids_file_path)

    if pos_dictionary is None:
        print("Failed to extract words. Exiting.")
        return

    # Save the result to a new JSON file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(pos_dictionary, f, indent=4)
        print(f"\nWords by part of speech have been extracted and saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving output file: {e}")


# Example usage
if __name__ == "__main__":
    main()

    # Alternative direct usage:
    # file_path = '/captions_avail_formatted.json'
    # video_ids_path = 'video_ids.txt'  # Optional: set to None to process all videos
    # pos_dictionary = extract_words_by_pos(file_path, video_ids_path)