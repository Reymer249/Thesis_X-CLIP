import json


def transform_json(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Transform the data to keep only the first sentence
    transformed_data = {}
    for key, sentences in data.items():
        if sentences.get("0", None) is None:
            transformed_data[key] = dict()
        else:
            transformed_data[key] = {
                "0": sentences["0"]
            }

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)


# Example usage
pos = "adverb"
input_file = f'/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_chen_provided/Chen_filtered/filtered_vatex1k5_{pos}_RE20.json'  # Replace with your input file path
output_file = f'/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_chen_provided/brittleness/filtered_vatex1k5_{pos}_RE20_brit.json'  # Replace with your desired output file path
transform_json(input_file, output_file)

print(f"Transformation complete. Output saved to {output_file}")