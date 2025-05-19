import json
import sys


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
    # Check if a filename was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicates.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]

    # Set output filename, default to "deduplicated_output.json" if not provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else "deduplicated_output.json"

    try:
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process the data to remove duplicates
        cleaned_data = remove_duplicates(data)

        # Save the cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

        # Print statistics
        original_counts = {id_key: len(paraphrases) for id_key, paraphrases in data.items()}
        cleaned_counts = {id_key: len(paraphrases) for id_key, paraphrases in cleaned_data.items()}

        total_original = sum(original_counts.values())
        total_cleaned = sum(cleaned_counts.values())

        print(f"Original data contained {total_original} total paraphrases.")
        print(f"After deduplication, {total_cleaned} unique paraphrases remain.")
        print(f"Removed {total_original - total_cleaned} duplicates.")
        print(f"Cleaned data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()