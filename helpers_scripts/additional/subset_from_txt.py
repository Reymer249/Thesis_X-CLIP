"""
    Random Line Selection Utility

    This module provides functionality to randomly select a specified percentage of lines
    from a text file and write the selected lines to a new output file. It's useful for
    creating random samples from large datasets, such as selecting a subset of video IDs,
    user records, or any line-based data for testing or analysis purposes.

    The script maintains the original formatting of each line and ensures no duplicate
    lines are selected from the input file. The selection is performed without replacement,
    meaning each line from the input can only appear once in the output.
"""
import random
import sys


def random_selection(input_file_path, output_file_path, percentage=20):
    """
    Randomly select a percentage of lines from input file and write to output file.

    Args:
        input_file_path (str): Path to the input text file.
        output_file_path (str): Path to the output text file.
        percentage (int): Percentage of lines to select (default: 20).
    """
    try:
        # Read all lines from input file
        with open(input_file_path, 'r') as input_file:
            all_lines = input_file.readlines()

        # Calculate how many lines to select
        total_lines = len(all_lines)
        lines_to_select = int(total_lines * percentage / 100)

        # Randomly select lines
        selected_lines = random.sample(all_lines, lines_to_select)

        # Write selected lines to output file
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(selected_lines)

        print(f"Successfully selected {lines_to_select} lines ({percentage}% of {total_lines}) "
              f"from '{input_file_path}' and wrote them to '{output_file_path}'.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python random_selection.py <input_file> <output_file> [percentage]")
        print("Example: python random_selection.py video_ids.txt selected_ids.txt 20")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Use provided percentage or default to 20%
    percentage = 20
    if len(sys.argv) > 3:
        try:
            percentage = int(sys.argv[3])
            if percentage <= 0 or percentage > 100:
                print("Error: Percentage must be between 1 and 100.")
                sys.exit(1)
        except ValueError:
            print("Error: Percentage must be a number.")
            sys.exit(1)

    random_selection(input_file, output_file, percentage)