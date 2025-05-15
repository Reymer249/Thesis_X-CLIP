import json
import sys


def count_instances(json_file_path):
    """
    Count the number of instances (keys) in a JSON file.

    Args:
        json_file_path (str): Path to the JSON file

    Returns:
        int: Number of instances in the JSON file
    """
    try:
        # Open and load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Count the number of keys (instances) in the JSON object
        num_instances = len(data)

        print(f"Total number of instances in the JSON file: {num_instances}")
        return num_instances

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        count_instances(json_file_path)
    else:
        print("Please provide the path to the JSON file as a command-line argument.")
        print("Usage: python count_json_instances.py path/to/your/file.json")