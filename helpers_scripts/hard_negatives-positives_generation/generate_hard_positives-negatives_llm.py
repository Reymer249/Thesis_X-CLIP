import json
import re
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard positives using LLM.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--video_ids_file', help='Path to TXT file containing video IDs to process (optional)')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard positives to generate per caption')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='LLM model to use')
    parser.add_argument('--gen_hard_neg', action="store_true", default=False)
    return parser.parse_args()


def generate_hard_positives_with_llm(caption: str, num_sentences: int, pipe, do_hard_neg: bool):
    """Generate hard positives for a caption using LLM"""

    # Create the prompt
    if do_hard_neg:
        prompt = [
            {"role": "user",
             "content": f"I will give you a sentence. Generate {num_sentences} hard negative sentences for it. "
                        f"A hard negative sentence is very similar in wording and structure to the original, but the "
                        f"meaning is different or opposite. Start by changing key words to antonyms or contrasting "
                        f"terms, or modifying the actions to contradict the original meaning. Keep the sentences "
                        f"fluent and grammatically correct.\n\n"
                        f"Example:\n"
                        f"Input: A man is hiking.\n"
                        f"Output:\n"
                        f"A woman is hiking.\n"
                        f"A female is hiking.\n"
                        f"A man is sitting.\n"
                        f"A man is lying.\n"
                        f"...\n"
                        f"Now generate hard negatives for this sentence:\n"
                        f"{caption}"
             }
        ]
    else:
        prompt = [
            {"role": "user",
             "content": f"Generate {num_sentences} different paraphrases of the following sentence that retain the same"
                        f" meaning but use different wording. Only output the paraphrases, one per line, without any "
                        f"additional text.\n\nSentence: {caption}"}
        ]

    # Generate paraphrases
    try:
        output = pipe(prompt, max_new_tokens=2048, return_full_text=False)
        paraphrases_text = output[0]["generated_text"]

        # Extract just the assistant's response
        if "assistant" in paraphrases_text.lower():
            # Find where the assistant's response begins
            assistant_idx = paraphrases_text.lower().find("assistant")
            # Get everything after "assistant" or "assistant:"
            response_text = paraphrases_text[assistant_idx:].split(":", 1)
            if len(response_text) > 1:
                paraphrases_text = response_text[1].strip()
            else:
                paraphrases_text = paraphrases_text[assistant_idx:].strip()

        # Split by newline and filter empty lines
        paraphrases = [p.strip() for p in paraphrases_text.split('\n') if p.strip()]

        # Remove any numbered prefixes like "1. " or "1) "
        paraphrases = [re.sub(r'^\d+[\.\)]\s*', '', p) for p in paraphrases]

        # Remove quotes if present
        paraphrases = [p.strip('"\'') for p in paraphrases]

        # Deduplicate and remove the original caption
        paraphrases = list(set(p for p in paraphrases if p.lower() != caption.lower()))

        # Limit to the requested number
        paraphrases = paraphrases[:num_sentences]

        # Format as [paraphrase, "paraphrase"] to maintain compatibility with original format
        # but without the POS tag since we're not using that logic anymore
        return [[p, "paraphrase"] for p in paraphrases]

    except Exception as e:
        print(f"Error generating paraphrases for '{caption}': {e}")
        return []


def plot_distribution(distribution, num_sentences, type_of_sentences):
    """Plot distribution of number of generated sentences"""
    plt.figure(figsize=(10, 6))
    x = np.arange(num_sentences + 1)  # 0 to num_sentences
    plt.bar(x, distribution)
    plt.xlabel('Number of Generated Hard Positives')
    plt.ylabel('Count')
    plt.title('Distribution of Generated Hard Positives per Caption')
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'hard_{type_of_sentences}_distribution_llm.png')
    plt.close()


def load_video_ids(file_path):
    """Load video IDs from a text file, one ID per line"""
    if not file_path:
        return None

    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace
            video_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(video_ids)} video IDs from {file_path}")
        return set(video_ids)  # Convert to set for faster lookups
    except Exception as e:
        print(f"Error loading video IDs file: {e}")
        return None


def main():
    args = parse_arguments()

    # Initialize the LLM pipeline
    print(f"Loading LLM model: {args.model_name}")
    pipe = pipeline("text-generation", model=args.model_name, temperature=0.6)

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Load video IDs if file is provided
    target_video_ids = load_video_ids(args.video_ids_file)

    hard_positives = {}

    # Initialize distribution array
    distribution = np.zeros(args.num_sentences + 1, dtype=int)

    # Process each video and its captions
    for video_id, captions in tqdm(captions_data.items()):
        # Skip videos not in our target list if we have a target list
        if target_video_ids is not None and video_id not in target_video_ids:
            continue

        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"

            # Generate hard sentences for this caption using LLM
            sentences = generate_hard_positives_with_llm(caption, args.num_sentences, pipe, args.gen_hard_neg)

            # Update distribution count
            count = min(len(sentences), args.num_sentences)
            distribution[count] += 1

            # Only add if we have any sentences
            if sentences:
                hard_positives[key] = sentences

    # Save distribution to pickle file
    sentences_type = "negatives" if args.gen_hard_neg else "positives"

    with open(f'hard_{sentences_type}_distribution_llm.pkl', 'wb') as f:
        pickle.dump(distribution, f)

    # Plot distribution
    plot_distribution(distribution, args.num_sentences, sentences_type)

    # Write output
    output_file = f"hard_{sentences_type}_{args.num_sentences}_llm.json"
    with open(output_file, 'w') as f:
        json.dump(hard_positives, f, indent=4)

    print(f"Generated hard sentences saved to {output_file}")
    print(
        f"Distribution saved to hard_{sentences_type}_distribution_llm.pkl "
        f"and hard_{sentences_type}_distribution_llm.png"
    )
    print(f"Distribution: {distribution}")


if __name__ == "__main__":
    main()