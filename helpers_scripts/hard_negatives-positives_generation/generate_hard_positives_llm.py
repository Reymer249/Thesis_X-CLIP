import json
import random
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard positives using LLM.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard positives to generate per caption')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='DeepSeek model to use')
    return parser.parse_args()


def generate_hard_positives_with_llm(caption, num_sentences, pipe):
    """Generate hard positives for a caption using DeepSeek LLM"""

    # Create the prompt
    prompt = [
        {"role": "user",
         "content": f"Generate {num_sentences} different paraphrases of the following sentence that retain the same"
                    f" meaning but use different wording. Only output the paraphrases, one per line, without any "
                    f"additional text.\n\nSentence: {caption}"}
    ]

    # Generate paraphrases
    try:
        output = pipe(prompt, max_new_tokens=1024, return_full_text=False)
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
    plt.savefig('hard_positives_distribution_llm.png')
    plt.close()


def main():
    args = parse_arguments()

    # Initialize the DeepSeek pipeline
    print(f"Loading DeepSeek model: {args.model_name}")
    pipe = pipeline("text-generation", model=args.model_name, temperature=0.6)

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

            # Generate hard positives for this caption using LLM
            positives = generate_hard_positives_with_llm(caption, args.num_sentences, pipe)

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
    output_file = "hard_positives.json"
    with open(output_file, 'w') as f:
        json.dump(hard_positives, f, indent=4)

    print(f"Generated hard positives saved to {output_file}")
    print(f"Distribution saved to hard_positives_distribution.pkl and hard_positives_distribution.png")
    print(f"Distribution: {distribution}")


if __name__ == "__main__":
    import re  # Added import for regex

    main()