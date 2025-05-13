import json
import re
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import pipeline
from datasets import Dataset

ESTIMATED_NUM_WORDS_PER_SENTENCE = 20
# 1 word ≈ 1.3 to 1.5 tokens (for English text, using models like GPT or BERT).
NUM_TOKENS = ESTIMATED_NUM_WORDS_PER_SENTENCE * 1.5


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate hard positives using LLM.')
    parser.add_argument('--captions_file', help='Path to JSON file with video captions')
    parser.add_argument('--video_ids_file', help='Path to TXT file containing video IDs to process (optional)')
    parser.add_argument('--num_sentences', type=int, default=20,
                        help='Maximum number of hard positives to generate per caption')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='LLM model to use')
    parser.add_argument('--gen_hard_neg', action="store_true", default=False)
    parser.add_argument('--max_retries', type=int, default=10,
                        help='Maximum number of generation attempts per caption')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing captions')
    return parser.parse_args()


def create_prompt(caption, request_count, do_hard_neg):
    """Create the prompt for generation"""
    if do_hard_neg:
        return [
            {"role": "user",
             "content": f"I will give you a sentence. Generate {request_count} hard negative sentences for it. "
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
        return [
            {"role": "user",
             "content": f"Generate {request_count} different paraphrases of the following sentence that retain the same"
                        f" meaning but use different wording. Only output the paraphrases, one per line, without any "
                        f"additional text, but with numbering. Stop after {request_count}.\n\nSentence: {caption}"}
        ]


def process_generated_text(generated_text, caption):
    """Process the generated text to extract paraphrases"""
    paraphrases_text = generated_text

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

    # Filter out duplicates and original caption
    return [p for p in paraphrases if p.lower() != caption.lower()]


def generate_hard_positives_batch(captions_data, video_ids, num_sentences, pipe, do_hard_neg, max_retries,
                                  batch_size=8):
    """Generate hard positives for captions in batches"""

    hard_positives = {}
    distribution = np.zeros(num_sentences + 1, dtype=int)

    # Prepare data for batch processing
    batch_data = []
    keys = []

    for video_id, captions in captions_data.items():
        # Skip videos not in our target list if we have a target list
        if video_ids is not None and video_id not in video_ids:
            continue

        for i, caption in enumerate(captions):
            key = f"{video_id}#{i}"
            keys.append(key)
            request_count = min(num_sentences * 2, 50)  # Same calculation as before
            batch_data.append({
                'caption': caption,
                'request_count': request_count,
                'do_hard_neg': do_hard_neg
            })

    # Create a dataset for batch processing
    dataset = Dataset.from_dict({
        'caption': [item['caption'] for item in batch_data],
        'request_count': [item['request_count'] for item in batch_data],
        'do_hard_neg': [item['do_hard_neg'] for item in batch_data],
        'key': keys
    })

    # Function to create prompts for map operation
    def prepare_prompts(examples):
        prompts = [create_prompt(caption, req_count, do_neg)
                   for caption, req_count, do_neg in zip(examples['caption'],
                                                         examples['request_count'],
                                                         examples['do_hard_neg'])]
        return {'prompts': prompts}

    # Map to create prompts
    dataset = dataset.map(prepare_prompts, batched=True)

    all_results = {}
    attempt_counter = {key: 0 for key in keys}

    # Retry logic for items that don't have enough results
    for retry in range(max_retries):
        # Filter dataset to only include items that need more sentences
        if retry > 0:
            # Get only keys that need more processing
            keys_needing_more = [key for key in keys
                                 if key in all_results
                                 and len(all_results[key]) < num_sentences
                                 and attempt_counter[key] < max_retries]

            if not keys_needing_more:
                break

            # Create a new dataset with just those items
            retry_data = []
            retry_keys = []

            for key in keys_needing_more:
                video_id, idx = key.split('#')
                idx = int(idx)
                caption = captions_data[video_id][idx]
                attempt_counter[key] += 1

                request_count = min((num_sentences - len(all_results[key])) * 2, 50)
                retry_data.append({
                    'caption': caption,
                    'request_count': request_count,
                    'do_hard_neg': do_hard_neg
                })
                retry_keys.append(key)

            # Create new dataset for retry
            dataset = Dataset.from_dict({
                'caption': [item['caption'] for item in retry_data],
                'request_count': [item['request_count'] for item in retry_data],
                'do_hard_neg': [item['do_hard_neg'] for item in retry_data],
                'key': retry_keys
            })

            # Map to create prompts again
            dataset = dataset.map(prepare_prompts, batched=True)

        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Retry {retry + 1}/{max_retries}"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))

            try:
                # Generate text for the batch
                outputs = pipe(
                    batch['prompts'],
                    max_new_tokens=int(NUM_TOKENS * num_sentences),
                    return_full_text=False,
                    batch_size=batch_size
                )

                # Process each output
                for j, output in enumerate(outputs):
                    key = batch['key'][j]
                    caption = batch['caption'][j]
                    generated_text = output[0]["generated_text"]

                    # Process the generated text
                    new_sentences = process_generated_text(generated_text, caption)

                    # Add to results, avoiding duplicates
                    if key not in all_results:
                        all_results[key] = [[s, "paraphrase"] for s in new_sentences]
                    else:
                        current_texts = [item[0].lower() for item in all_results[key]]
                        for s in new_sentences:
                            if s.lower() not in current_texts and len(all_results[key]) < num_sentences:
                                all_results[key].append([s, "paraphrase"])

            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue

    # Update distribution and limit results to num_sentences
    for key in keys:
        sentences = all_results.get(key, [])
        count = min(len(sentences), num_sentences)
        distribution[count] += 1

        # Limit to requested number of sentences
        if key in all_results:
            all_results[key] = all_results[key][:num_sentences]

    # Filter out empty results
    hard_positives = {k: v for k, v in all_results.items() if v}

    return hard_positives, distribution


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
    # Load tokenizer and model separately to set padding_side
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Create pipeline with the configured tokenizer and model
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.6)

    # Load captions
    with open(args.captions_file, 'r') as f:
        captions_data = json.load(f)

    # Load video IDs if file is provided
    target_video_ids = load_video_ids(args.video_ids_file)

    # Process captions in batches
    hard_positives, distribution = generate_hard_positives_batch(
        captions_data,
        target_video_ids,
        args.num_sentences,
        pipe,
        args.gen_hard_neg,
        args.max_retries,
        args.batch_size
    )

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