import json
import random
import argparse
import logging
from pathlib import Path

def load_json_file(filepath):
    """Load and return JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {filepath}.")
        return None

def generate_examples(captions_file, pos_negatives_file, pos_negatives_file_20, 
                     pos_negatives_file_5, pos_positives_file, pos_positives_file_20,
                     pos_positives_file_5, llm_negatives_file, llm_positives_file,
                     n_examples=5, verbose=False):
    """
    Generate n random examples matching captions with their hard negatives/positives.
    
    Args:
        captions_file: Path to the main captions JSON file
        pos_negatives_file: Path to POS-generated hard negatives (100)
        pos_negatives_file_20: Path to POS-generated hard negatives (20)
        pos_negatives_file_5: Path to POS-generated hard negatives (5)
        pos_positives_file: Path to POS-generated hard positives (40)
        pos_positives_file_20: Path to POS-generated hard positives (20)
        pos_positives_file_5: Path to POS-generated hard positives (5)
        llm_negatives_file: Path to LLM-generated hard negatives
        llm_positives_file: Path to LLM-generated hard positives
        n_examples: Number of examples to generate
        verbose: If True, additionally print examples for sets of 20 and 5
    """
    
    # Load all JSON files
    captions = load_json_file(captions_file)
    logging.info(f"Loaded {len(captions)} captions from {captions_file}")
    pos_negatives = load_json_file(pos_negatives_file)
    logging.info(f"Loaded {len(pos_negatives)} POS hard negatives from {pos_negatives_file}")
    pos_negatives_20 = load_json_file(pos_negatives_file_20)
    logging.info(f"Loaded {len(pos_negatives_20)} POS hard negatives (20) from {pos_negatives_file_20}")
    pos_negatives_5 = load_json_file(pos_negatives_file_5)
    logging.info(f"Loaded {len(pos_negatives_5)} POS hard negatives (5) from {pos_negatives_file_5}")
    pos_positives = load_json_file(pos_positives_file)
    logging.info(f"Loaded {len(pos_positives)} POS hard positives from {pos_positives_file}")
    pos_positives_20 = load_json_file(pos_positives_file_20)
    logging.info(f"Loaded {len(pos_positives_20)} POS hard positives (20) from {pos_positives_file_20}")
    pos_positives_5 = load_json_file(pos_positives_file_5)
    logging.info(f"Loaded {len(pos_positives_5)} POS hard positives (5) from {pos_positives_file_5}")
    llm_negatives = load_json_file(llm_negatives_file)
    logging.info(f"Loaded {len(llm_negatives)} LLM hard negatives from {llm_negatives_file}")
    llm_positives = load_json_file(llm_positives_file)
    logging.info(f"Loaded {len(llm_positives)} LLM hard positives from {llm_positives_file}")
    
    # Check if all files loaded successfully
    files_to_check = [captions, pos_negatives, pos_negatives_20, pos_negatives_5,
                     pos_positives, pos_positives_20, pos_positives_5,
                     llm_negatives, llm_positives]
    if not all(files_to_check):
        print("Error: Failed to load one or more JSON files.")
        return
    
    print(f"Generating {n_examples} random examples:\n")
    print("=" * 80)
    
    for i in range(n_examples):
        # Select random video and caption
        video_id = random.choice(list(captions.keys()))
        caption_list = captions[video_id]
        caption_idx = random.randint(0, len(caption_list) - 1)
        selected_caption = caption_list[caption_idx]
        
        # Create the key for looking up hard examples
        lookup_key = f"{video_id}#{caption_idx}"
        
        print(f"\nExample {i + 1}:")
        print(f"Video ID: {video_id}")
        print(f"Original Caption: {selected_caption}")
        
                # Get POS-generated examples (all variants)
        print("\n--- POS-Generated Examples ---")
        if verbose:
            get_random_example(pos_negatives, lookup_key, "POS Hard Negative (100)")
            get_random_example(pos_negatives_20, lookup_key, "POS Hard Negative (20)")
            get_random_example(pos_negatives_5, lookup_key, "POS Hard Negative (5)")

            get_random_example(pos_positives, lookup_key, "POS Hard Positive (40)")
            get_random_example(pos_positives_20, lookup_key, "POS Hard Positive (20)")
            get_random_example(pos_positives_5, lookup_key, "POS Hard Positive (5)")
        else:
            get_random_example(pos_negatives, lookup_key, "POS Hard Negative")
            get_random_example(pos_positives, lookup_key, "POS Hard Positive")

        # Get LLM-generated examples
        print("\n--- LLM-Generated Examples ---")
        get_random_example(llm_negatives, lookup_key, "LLM Hard Negative")
        get_random_example(llm_positives, lookup_key, "LLM Hard Positive")

        # Separator for clarity
        print("-" * 40)

def get_random_example(data_dict, lookup_key, example_type):
    """
    Get a random example from the data dictionary for the given key.
    
    Args:
        data_dict: Dictionary containing the hard examples
        lookup_key: Key to look up (format: video_id#caption_number)
        example_type: Type of example for display purposes
    
    Returns:
        Random example text or None if not found
    """
    if lookup_key in data_dict:
        examples_list = data_dict[lookup_key]
        if examples_list:
            # Select random example from the list
            random_example = random.choice(examples_list)
            example_text = random_example[0]  # First element is the text
            
            print(f"{example_type}: {example_text}")
            return example_text
    
    print(f"{example_type}: [NOT FOUND]")
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate random caption examples with hard negatives/positives')
    
    # Change these to optional arguments by adding a hyphen prefix
    parser.add_argument('--captions_file', help='Path to the main captions JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/captions_avail_train_020.json')
    parser.add_argument('--pos_negatives_file', help='Path to POS-generated hard negatives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_all_pos_100.json')
    parser.add_argument('--pos_negatives_file_20', help='Path to POS-generated hard negatives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_all_pos_20.json')
    parser.add_argument('--pos_negatives_file_5', help='Path to POS-generated hard negatives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_all_pos_5.json')
    parser.add_argument('--pos_positives_file', help='Path to POS-generated hard positives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_all_pos_40.json')
    parser.add_argument('--pos_positives_file_20', help='Path to POS-generated hard positives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_all_pos_20.json')
    parser.add_argument('--pos_positives_file_5', help='Path to POS-generated hard positives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_all_pos_5.json')
    parser.add_argument('--llm_negatives_file', help='Path to LLM-generated hard negatives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_llm_final.json')
    parser.add_argument('--llm_positives_file', help='Path to LLM-generated hard positives JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_llm_final.json')

    parser.add_argument('-n', '--num_examples', type=int, default=25,
                       help='Number of examples to generate (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    random.seed(23)  # For reproducibility
    
    generate_examples(
        args.captions_file,
        args.pos_negatives_file,
        args.pos_negatives_file_20,
        args.pos_negatives_file_5,
        args.pos_positives_file,
        args.pos_positives_file_20,
        args.pos_positives_file_5,
        args.llm_negatives_file,
        args.llm_positives_file,
        args.num_examples,
        args.verbose
    )

if __name__ == "__main__":
    main()