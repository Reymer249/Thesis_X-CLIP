"""
    This scripts extracts captions fot the specific videos from the *RE20.json file.
"""
import json

def extract_captions_for_video(json_file_path, target_video_id):
    """
    Extract all captions for a specific video ID from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        target_video_id (str): Video ID to extract captions for
        
    Returns:
        dict: A dictionary of captions where keys are the caption numbers and values are the caption content
    """
    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' contains invalid JSON.")
        return {}
    
    # Extract captions for the target video ID
    captions = {}
    for key, value in data.items():
        # Check if key matches the expected format: [video_id]#[caption_number]
        if '#' in key:
            video_id, caption_number = key.split('#', 1)
            if video_id == target_video_id:
                captions[caption_number] = value
    
    return captions

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path and video ID
    file_path = "/vol/home/s3705609/Desktop/thesis_code/filtered_json_files/filtered_vatex1k5_adjective_RE20.json"
    video_id = "WsEfaI-2azs_000024_000034"
    
    result = extract_captions_for_video(file_path, video_id)
    
    if result:
        print(f"Found {len(result)} captions for video ID: {video_id}")
        for caption_num, caption_text in result.items():
            print(f"Caption #{caption_num}: {caption_text}")
    else:
        print(f"No captions found for video ID: {video_id}")