"""
    Before myy research I was provided with a *_RE20.json files. These are files with hard negative sentences. As I don't
    have all the videos described in these files, I had to filter them to leave only the descriptions for the
    videos I have. This file does that.
"""
import json
import re
import os
from typing import Set, Dict, Any, Tuple


def extract_video_ids_from_json(json_data: Dict[str, Any]) -> Set[str]:
    """Extract unique video IDs from JSON keys with format '[video_id]#[caption-number]'."""
    video_ids = set()
    for key in json_data.keys():
        # Extract the video ID part before the '#' character
        match = re.match(r'([^#]+)#', key)
        if match:
            video_id = match.group(1)
            video_ids.add(video_id)
    return video_ids


def read_video_ids_from_txt(file_path: str) -> Set[str]:
    """Read video IDs from a text file with one ID per line."""
    with open(file_path, 'r') as f:
        # Strip whitespace and filter out empty lines
        return set(line.strip() for line in f if line.strip())


def process_json_file(json_file_path: str, txt_video_ids: Set[str], output_dir: str) -> Tuple[int, int, int]:
    """
    Process a single JSON file:
    1. Find video IDs present in JSON but not in txt file
    2. Create a new JSON file without those missing videos
    3. Return statistics: (total entries, unique video IDs, deleted video IDs)
    """
    try:
        # Load JSON data
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        # Get total number of entries
        total_entries = len(json_data)

        # Extract all video IDs from JSON
        all_json_video_ids = extract_video_ids_from_json(json_data)
        total_unique_ids = len(all_json_video_ids)

        # Find missing video IDs
        missing_video_ids = all_json_video_ids - txt_video_ids

        # Create a new JSON without entries for missing video IDs
        filtered_json_data = {}
        for key, value in json_data.items():
            # Extract video ID from the key
            match = re.match(r'([^#]+)#', key)
            if match and match.group(1) not in missing_video_ids:
                filtered_json_data[key] = value

        # Create output filename
        base_name = os.path.basename(json_file_path)
        output_file_path = os.path.join(output_dir, f"filtered_{base_name}")

        # Write filtered data to new JSON file
        with open(output_file_path, 'w') as f:
            json.dump(filtered_json_data, f, indent=2)

        return total_entries, total_unique_ids, len(missing_video_ids)

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return 0, 0, 0


def main():
    # File paths
    base = "/vol/home/s3705609/Desktop/data_vatex/splits_txt/"
    txt_file_path = base + "vatex_val_avail.txt"
    json_files = [
        "vatex1k5_noun_RE20.json",
        "vatex1k5_adjective_RE20.json",
        "vatex1k5_verb_RE20.json",
        "vatex1k5_adverb_RE20.json",
        "vatex1k5_preposition_RE20.json",
    ]
    json_files = [base + file for file in json_files]

    # Create output directory if it doesn't exist
    output_dir = "filtered_json_files"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load video IDs from text file
        txt_video_ids = read_video_ids_from_txt(txt_file_path)
        print(f"Found {len(txt_video_ids)} video IDs in the text file")
        print("-" * 80)

        # Process each JSON file
        total_entries = 0
        total_unique_ids = 0
        total_deleted_ids = 0

        print(f"{'File':<30} {'Total Entries':<15} {'Unique IDs':<15} {'Deleted IDs':<15}")
        print("-" * 80)

        for json_file in json_files:
            entries, unique_ids, deleted_ids = process_json_file(json_file, txt_video_ids, output_dir)

            # Update totals
            total_entries += entries
            total_unique_ids += unique_ids
            total_deleted_ids += deleted_ids

            # Print statistics for this file
            print(f"{json_file:<30} {entries:<15} {unique_ids:<15} {deleted_ids:<15}")

        # Print summary
        print("-" * 80)
        print(f"{'TOTAL:':<30} {total_entries:<15} {total_unique_ids:<15} {total_deleted_ids:<15}")
        print("-" * 80)
        print(f"Filtered JSON files saved to {output_dir}/ directory")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()