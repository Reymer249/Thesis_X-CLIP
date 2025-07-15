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

def extract_pos_examples(examples_list, target_pos):
    """
    Extract examples that contain the target part of speech in comments.
    
    Args:
        examples_list: List of examples for a given caption
        target_pos: Target part of speech (e.g., "VERB", "ADJ", "PREP", "ADV", "NOUN")
    
    Returns:
        List of examples that contain the target POS
    """
    pos_examples = []
    for example in examples_list:
        if len(example) >= 2:  # Make sure we have both text and metadata
            example_text = example[0]
            metadata = example[1] if len(example) > 1 else ""
            
            # Check if the target POS is mentioned in the metadata/comments
            if target_pos in str(metadata):
                pos_examples.append(example_text)
    
    return pos_examples

def generate_pos_examples(captions_file, pos_negatives_file, pos_positives_file, n_examples=5):
    """
    Generate n random examples showing original caption and POS-based hard negatives/positives.
    
    Args:
        captions_file: Path to the main captions JSON file
        pos_negatives_file: Path to POS-generated hard negatives (100)
        pos_positives_file: Path to POS-generated hard positives (40)
        n_examples: Number of examples to generate
    """
    
    # Load JSON files
    captions = load_json_file(captions_file)
    if not captions:
        return
    logging.info(f"Loaded {len(captions)} captions from {captions_file}")
    
    pos_negatives = load_json_file(pos_negatives_file)
    if not pos_negatives:
        return
    logging.info(f"Loaded {len(pos_negatives)} POS hard negatives from {pos_negatives_file}")
    
    pos_positives = load_json_file(pos_positives_file)
    if not pos_positives:
        return
    logging.info(f"Loaded {len(pos_positives)} POS hard positives from {pos_positives_file}")
    
    # Parts of speech to extract examples for
    pos_tags = ["VERB", "ADJ", "PREP", "ADV", "NOUN"]
    
    print(f"Generating {n_examples} random examples with POS-based hard negatives/positives:\n")
    print("=" * 100)
    
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
        print("-" * 60)
        
        # Process hard negatives
        print("POS Hard Negatives (100):")
        if lookup_key in pos_negatives:
            negatives_list = pos_negatives[lookup_key]
            for pos_tag in pos_tags:
                pos_examples = extract_pos_examples(negatives_list, pos_tag)
                if pos_examples:
                    example = random.choice(pos_examples)
                    print(f"  {pos_tag}: {example}")
                else:
                    print(f"  {pos_tag}: [NOT FOUND]")
        else:
            print(f"  [NO NEGATIVES FOUND FOR {lookup_key}]")
        
        print()
        
        # Process hard positives
        print("POS Hard Positives (40):")
        if lookup_key in pos_positives:
            positives_list = pos_positives[lookup_key]
            for pos_tag in pos_tags:
                pos_examples = extract_pos_examples(positives_list, pos_tag)
                if pos_examples:
                    example = random.choice(pos_examples)
                    print(f"  {pos_tag}: {example}")
                else:
                    print(f"  {pos_tag}: [NOT FOUND]")
        else:
            print(f"  [NO POSITIVES FOUND FOR {lookup_key}]")
        
        # Separator for clarity
        print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description='Generate random caption examples with POS-based hard negatives/positives')
    
    parser.add_argument('--captions_file', help='Path to the main captions JSON file',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/captions_avail_train_020.json')
    parser.add_argument('--pos_negatives_file', help='Path to POS-generated hard negatives JSON file (100)',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_all_pos_100.json')
    parser.add_argument('--pos_positives_file', help='Path to POS-generated hard positives JSON file (40)',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_all_pos_40.json')
    parser.add_argument('-n', '--num_examples', type=int, default=5,
                       help='Number of examples to generate (default: 5)')
    
    args = parser.parse_args()
    random.seed(23)  # For reproducibility
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    generate_pos_examples(
        args.captions_file,
        args.pos_negatives_file,
        args.pos_positives_file,
        args.num_examples
    )

if __name__ == "__main__":
    main()