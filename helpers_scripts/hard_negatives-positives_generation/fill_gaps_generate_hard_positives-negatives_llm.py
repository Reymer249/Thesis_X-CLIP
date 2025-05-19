import datetime
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

ESTIMATED_NUM_WORDS_PER_SENTENCE = 20
# 1 word ≈ 1.3 to 1.5 tokens (for English text, using models like GPT or BERT).
NUM_TOKENS = ESTIMATED_NUM_WORDS_PER_SENTENCE * 1.5


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate missing hard positives/negatives using LLM.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--generated_file', help='Path to previously generated JSON file with hard sentences')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard sentences to generate per caption')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='Model to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--gen_hard_neg', action='store_true',
                        help='If set, we generate hard negatives. Otherwise - hard positives')
    return parser.parse_args()


def create_prompt(caption, num_sentences, do_hard_neg):
    """Create prompt for generating paraphrases"""
    if do_hard_neg:
        content = f"I will give you a sentence. Generate {num_sentences} hard negative sentences for it. " \
                  f"A hard negative sentence is very similar in wording and structure to the original, but the " \
                  f"meaning is different or opposite. Start by changing key words to antonyms or contrasting " \
                  f"terms, or modifying the actions to contradict the original meaning. Keep the sentences " \
                  f"fluent and grammatically correct.\n\n" \
                  f"Example:\n" \
                  f"Input: A man is hiking.\n" \
                  f"Output:\n" \
                  f"A woman is hiking.\n" \
                  f"A female is hiking.\n" \
                  f"A man is sitting.\n" \
                  f"A man is lying.\n" \
                  f"...\n" \
                  f"Now generate hard negatives for this sentence:\n" \
                  f"{caption}"
    else:
        content = f"Generate {num_sentences} different paraphrases of the following sentence that retain the same " \
                  f"meaning but use different wording. Only output the paraphrases, one per line, without any " \
                  f"additional text.\n\nSentence: {caption}"
    return [
        {
            "role": "user",
            "content": content
        }
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


def find_missing_sentences(captions_data, generated_data, target_num_sentences):
    """Find captions that need additional sentences to reach the target number"""
    missing_captions = {}
    captions_map = {}  # Map of key to original caption

    # Create a dictionary to map keys to their captions
    for video_id, captions in captions_data.items():
        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"
            captions_map[key] = caption

    # Find captions with fewer than target sentences
    for key, caption in captions_map.items():
        existing_count = 0
        if key in generated_data:
            existing_count = len(generated_data[key])

        if existing_count < target_num_sentences:
            sentences_needed = target_num_sentences - existing_count
            missing_captions[key] = {
                "caption": caption,
                "existing_count": existing_count,
                "sentences_needed": sentences_needed
            }

    return missing_captions, captions_map


def generate_missing_sentences_batch(missing_captions, generated_data, num_sentences, pipe, batch_size=8,
                                     gen_hard_neg=False):
    """Generate the missing hard sentences for captions that have fewer than the target number"""

    # Prepare data for batched processing
    all_captions = []
    all_keys = []
    sentences_needed = []

    for key, data in missing_captions.items():
        all_captions.append(data["caption"])
        all_keys.append(key)
        sentences_needed.append(data["sentences_needed"])

    # Process in batches
    updated_data = generated_data.copy()
    distribution_update = np.zeros(num_sentences + 1, dtype=int)

    # Create generator for streaming input - only generate the number of sentences needed for each caption
    def caption_generator():
        for caption, needed in zip(all_captions, sentences_needed):
            yield create_prompt(caption, needed, gen_hard_neg)

    # Process batches with progress bar
    results = []
    batch_counter = 0
    for batch_results in tqdm(pipe(caption_generator(),
                                   max_new_tokens=NUM_TOKENS * num_sentences,
                                   batch_size=batch_size),
                              total=len(all_captions),
                              desc="Generating missing sentences"):
        results.append(batch_results)
        batch_counter += 1
        if batch_counter % 5 == 0:
            with open("tmp_results.pkl", "wb") as f:
                pickle.dump(results, f)
            with open("batch_counter.pkl", "wb") as b_f:
                pickle.dump(batch_counter, b_f)

    # Process results
    for i, result in tqdm(enumerate(results)):
        caption = all_captions[i]
        key = all_keys[i]
        needed = sentences_needed[i]

        try:
            paraphrases_text = result[0]["generated_text"][1]["content"]
            new_sentences = process_paraphrases(paraphrases_text, caption)

            # Limit to the requested number
            new_sentences = new_sentences[:needed]

            # Get existing sentences or initialize empty list
            existing_sentences = updated_data.get(key, [])

            # Combine existing and new sentences
            combined_sentences = existing_sentences + new_sentences

            # Update the data
            updated_data[key] = combined_sentences

            # Update distribution - count how many we have now
            final_count = len(combined_sentences)
            distribution_update[final_count] += 1

        except Exception as e:
            print(f"Error processing results for '{caption}': {e}")
            # If there was an error, keep the existing entries
            if key in updated_data:
                current_count = len(updated_data[key])
                distribution_update[current_count] += 1

    return updated_data, distribution_update


def plot_distribution(original_distribution, updated_distribution, num_sentences, sentences_type, timestamp):
    """Plot distribution of number of generated sentences before and after filling"""
    plt.figure(figsize=(12, 6))
    x = np.arange(num_sentences + 1)  # 0 to num_sentences

    width = 0.35
    plt.bar(x - width / 2, original_distribution, width, label='Before Filling')
    plt.bar(x + width / 2, updated_distribution, width, label='After Filling')

    plt.xlabel('Number of Generated Hard Sentences')
    plt.ylabel('Count')
    plt.title(f'Distribution of Generated Hard {sentences_type.capitalize()} per Caption')
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'hard_{sentences_type}_distribution_filled_{timestamp}.png')
    plt.close()


def calculate_original_distribution(generated_data, num_sentences):
    """Calculate the distribution of sentence count in the original data"""
    distribution = np.zeros(num_sentences + 1, dtype=int)

    for key, sentences in generated_data.items():
        count = min(len(sentences), num_sentences)
        distribution[count] += 1

    return distribution


def save_files(updated_data, updated_distribution, original_distribution, args):
    # Get timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine sentence type for naming
    sentences_type = "negatives" if args.gen_hard_neg else "positives"

    # Save the updated data
    output_file = f"hard_{sentences_type}_filled_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=4)

    # Save distribution to pickle file
    with open(f'hard_{sentences_type}_distribution_filled_{timestamp}.pkl', 'wb') as f:
        pickle.dump(updated_distribution, f)

    # Plot distribution
    plot_distribution(original_distribution, updated_distribution, args.num_sentences, sentences_type, timestamp)

    print(f"Updated data saved to {output_file}")
    print(f"Distribution saved to hard_{sentences_type}_distribution_filled_{timestamp}.pkl")
    print(f"Original distribution: {original_distribution}")
    print(f"Updated distribution: {updated_distribution}")


def remove_duplicates(data):
    """
    Remove duplicate paraphrases for each ID in the data.
    Duplicates are identified based on the exact string match of the paraphrase text.
    """
    cleaned_data = {}

    for id_key, paraphrases in data.items():
        # Use a set to track unique paraphrases
        seen_paraphrases = set()
        unique_paraphrases = []

        for paraphrase_pair in paraphrases:
            # The first element in the pair is the paraphrase text
            paraphrase_text = paraphrase_pair[0]

            # If we haven't seen this paraphrase before, add it to our result
            if paraphrase_text not in seen_paraphrases:
                seen_paraphrases.add(paraphrase_text)
                unique_paraphrases.append(paraphrase_pair)

        # Add the deduplicated list to our result
        cleaned_data[id_key] = unique_paraphrases

    return cleaned_data


def main():
    args = parse_arguments()

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Load previously generated data
    with open(args.generated_file, 'r') as f:
        generated_data = json.load(f)

    original_counts = {id_key: len(paraphrases) for id_key, paraphrases in generated_data.items()}
    generated_data = remove_duplicates(generated_data)
    cleaned_counts = {id_key: len(paraphrases) for id_key, paraphrases in generated_data.items()}

    total_original = sum(original_counts.values())
    total_cleaned = sum(cleaned_counts.values())

    print(f"Original data contained {total_original} total paraphrases.")
    print(f"After deduplication, {total_cleaned} unique paraphrases remain.")
    print(f"Removed {total_original - total_cleaned} duplicates.")

    # Initialize the LLM pipeline
    print(f"Loading LLM model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.6,
        device=0  # Use GPU if available
    )

    finished_flag = False
    updated_distribution, original_distribution = None, None

    while not finished_flag:
        # Find captions that need additional sentences
        missing_captions, captions_map = find_missing_sentences(captions_data, generated_data, args.num_sentences)

        if not missing_captions:
            print("No missing sentences found. All captions have the target number of sentences.")
            return

        print(f"Found {len(missing_captions)} captions that need additional sentences.")

        # Calculate original distribution
        original_distribution = calculate_original_distribution(generated_data, args.num_sentences)

        # Generate missing sentences
        updated_data, updated_distribution = generate_missing_sentences_batch(
            missing_captions,
            generated_data,
            args.num_sentences,
            pipe,
            batch_size=args.batch_size,
            gen_hard_neg=args.gen_hard_neg
        )

        generated_data = remove_duplicates(updated_data)
        original_counts = {id_key: len(paraphrases) for id_key, paraphrases in updated_data.items()}
        cleaned_counts = {id_key: len(paraphrases) for id_key, paraphrases in generated_data.items()}

        total_original = sum(original_counts.values())
        total_cleaned = sum(cleaned_counts.values())

        if total_original - total_cleaned == 0:  # so all the captions are unique
            finished_flag = True

    if generated_data is not None and updated_distribution is not None and original_distribution is not None:
        save_files(generated_data, updated_distribution, original_distribution, args)
    else:
        raise Exception("Some error occurred")


if __name__ == "__main__":
    main()