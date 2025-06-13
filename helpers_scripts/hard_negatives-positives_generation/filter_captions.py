"""
    Filters JSON file with video captions to include only videos listed in the txt file.
"""
import json
import os


def filter_json_with_video_ids(captions_json_path, video_ids_txt_path, output_json_path=None):
    """
    Filter JSON file with video captions to include only videos listed in the txt file.

    Args:
        captions_json_path (str): Path to the JSON file with video captions
        video_ids_txt_path (str): Path to the txt file with video IDs
        output_json_path (str, optional): Path for the output filtered JSON file.
                                         If None, will use "filtered_captions.json"

    Returns:
        dict: Filtered captions dictionary
    """
    # Set default output path if not provided
    if output_json_path is None:
        output_json_path = "filtered_captions.json"

    # Read video IDs from the txt file
    with open(video_ids_txt_path, 'r') as f:
        video_ids = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(video_ids)} video IDs from {video_ids_txt_path}")

    # Read the captions JSON file
    with open(captions_json_path, 'r') as f:
        captions_data = json.load(f)

    print(f"Loaded captions for {len(captions_data)} segments from {captions_json_path}")

    # Filter captions to include only videos in the txt file
    filtered_data = {}

    for segment_id, captions in captions_data.items():
        # Extract video ID from segment ID (assuming format: videoID_starttime_endtime)
        # video_id = segment_id.split('_')[0]

        if segment_id in video_ids:
            filtered_data[segment_id] = captions

    print(
        f"Filtered down to {len(filtered_data)} segments")

    # Save filtered data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered captions saved to {output_json_path}")

    return filtered_data


if __name__ == "__main__":
    # Prompt user for file paths
    captions_json_path = input("Captions path:")
    video_ids_txt_path = input("IDs path:")
    output_json_path = input("Output path:")

    if not output_json_path:
        output_json_path = "filtered_captions.json"

    # Run the filtering function
    filter_json_with_video_ids(captions_json_path, video_ids_txt_path, output_json_path)