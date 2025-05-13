import json
import random
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import pipeline
import re



ESTIMATED_NUM_WORDS_PER_SENTENCE = 20
# 1 word ≈ 1.3 to 1.5 tokens (for English text, using models like GPT or BERT).
NUM_TOKENS = ESTIMATED_NUM_WORDS_PER_SENTENCE * 1.5

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard positives using LLM.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard positives to generate per caption')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='Model to use')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    return parser.parse_args()


def create_prompt(caption, num_sentences):
    """Create prompt for generating paraphrases"""
    return [
        {"role": "user",
         "content": f"Generate {num_sentences} different paraphrases of the following sentence that retain the same"
                    f" meaning but use different wording. Only output the paraphrases, one per line, without any "
                    f"additional text.\n\nSentence: {caption}"}
    ]


def process_paraphrases(paraphrases_text, caption):
    """Process the raw output from the model into formatted paraphrases"""
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

    # Format as [paraphrase, "paraphrase"] to maintain compatibility with original format
    return [[p, "paraphrase"] for p in paraphrases]


def generate_hard_positives_batch(captions_data, num_sentences, pipe, batch_size=8):
    """Generate hard positives for multiple captions using batched processing"""

    # Prepare data for batched processing
    all_captions = []
    all_keys = []

    for video_id, captions in captions_data.items():
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"
            all_captions.append(caption)
            all_keys.append(key)

    # Process in batches
    hard_positives = {}
    distribution = np.zeros(num_sentences + 1, dtype=int)

    # Create generator for streaming input
    def caption_generator():
        for caption in all_captions:
            yield create_prompt(caption, num_sentences)

    # Process batches with progress bar
    results = []
    batch_counter = 0
    for batch_results in tqdm(pipe(caption_generator(),
                                   max_new_tokens=NUM_TOKENS*num_sentences,
                                   batch_size=batch_size),
                              total=len(all_captions),
                              desc="Generating paraphrases"):
        results.append(batch_results)
        batch_counter += 1
        if batch_counter % 5 == 0:
            with open("tmp_results.pkl", "wb") as f:
                pickle.dump(results, f)
            with open("batch_counter.pkl", "wb") as b_f:
                pickle.dump(batch_counter, b_f)

    # Process results
    for i, result in enumerate(results):
        caption = all_captions[i]
        key = all_keys[i]

        try:
            paraphrases_text = result[0]["generated_text"]
            positives = process_paraphrases(paraphrases_text, caption)

            # Limit to the requested number
            positives = positives[:num_sentences]

            # Update distribution count
            count = min(len(positives), num_sentences)
            distribution[count] += 1

            # Only add if we have any positives
            if positives:
                hard_positives[key] = positives

        except Exception as e:
            print(f"Error processing results for '{caption}': {e}")

    return hard_positives, distribution


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

    # Initialize the LLM pipeline with streaming and batching
    # Initialize the LLM pipeline
    print(f"Loading LLM model: {args.model_name}")
    # Load tokenizer and model separately to set padding_side
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(args.model_name)


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.6,
        device=1  # Use GPU if available
    )

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Generate hard positives using batched processing
    hard_positives, distribution = generate_hard_positives_batch(
        captions_data,
        args.num_sentences,
        pipe,
        batch_size=args.batch_size
    )

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
    main()