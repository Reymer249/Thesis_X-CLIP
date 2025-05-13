import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_sentence_lengths(json_file_path):
    # Load the JSON data from file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # List to store word counts for each sentence
    word_counts = []

    # Iterate through each video ID and its sentences
    for video_id, sentences in data.items():
        for sentence in sentences:
            # Count words in each sentence
            words = sentence.split()
            word_counts.append(len(words))

    # Calculate statistics
    if not word_counts:
        return {
            "total_sentences": 0,
            "mean_words": 0,
            "median_words": 0,
            "min_words": 0,
            "max_words": 0,
            "quartiles": {"q1": 0, "q2": 0, "q3": 0},
            "min_example": None,
            "max_example": None
        }

    mean_words = statistics.mean(word_counts)
    median_words = statistics.median(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)

    # Calculate quartiles
    q1 = np.percentile(word_counts, 25)
    q2 = np.percentile(word_counts, 50)  # same as median
    q3 = np.percentile(word_counts, 75)

    # Find example sentences for min and max
    min_example = None
    max_example = None

    for video_id, sentences in data.items():
        for sentence in sentences:
            words = sentence.split()
            if len(words) == min_words and min_example is None:
                min_example = sentence
            if len(words) == max_words and max_example is None:
                max_example = sentence
            if min_example is not None and max_example is not None:
                break

    return {
        "total_sentences": len(word_counts),
        "mean_words": mean_words,
        "median_words": median_words,
        "min_words": min_words,
        "max_words": max_words,
        "quartiles": {"q1": q1, "q2": q2, "q3": q3},
        "word_counts": word_counts,
        "min_example": min_example,
        "max_example": max_example
    }


def plot_distribution(word_counts):
    plt.figure(figsize=(12, 10))

    # Set up the figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Histogram
    ax1.hist(word_counts, bins=range(min(word_counts), max(word_counts) + 2),
             edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Words per Sentence')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # 2. Kernel Density Plot
    sns.kdeplot(word_counts, ax=ax2, fill=True)
    ax2.set_title('Kernel Density Estimation')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)

    # 3. Box Plot
    ax3.boxplot(word_counts, vert=False, widths=0.7, patch_artist=True)
    ax3.set_title('Box Plot of Words per Sentence')
    ax3.set_xlabel('Number of Words')
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sentence_length_distribution.png')
    plt.close()

    # Create a frequency distribution for the most common sentence lengths
    counter = Counter(word_counts)
    most_common = counter.most_common(10)

    # Bar chart for most common sentence lengths
    plt.figure(figsize=(10, 6))
    labels = [str(x[0]) for x in most_common]
    values = [x[1] for x in most_common]

    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.title('Most Common Sentence Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('common_sentence_lengths.png')
    plt.close()


# Example usage
if __name__ == "__main__":
    file_path = "/vol/home/s3705609/Desktop/data_vatex/splits_txt/captions_avail_formatted.json"
    results = analyze_sentence_lengths(file_path)

    print(f"Total sentences analyzed: {results['total_sentences']}")
    print(f"Mean words per sentence: {results['mean_words']:.2f}")
    print(f"Median words per sentence: {results['median_words']:.2f}")
    print(f"Minimum words in a sentence: {results['min_words']}")
    print(f"Maximum words in a sentence: {results['max_words']}")
    print(f"Quartiles:")
    print(f"  - Q1 (25%): {results['quartiles']['q1']:.2f}")
    print(f"  - Q2 (50%): {results['quartiles']['q2']:.2f}")
    print(f"  - Q3 (75%): {results['quartiles']['q3']:.2f}")
    print(f"\nExample of shortest sentence: \"{results['min_example']}\"")
    print(f"Example of longest sentence: \"{results['max_example']}\"")

    # Generate distribution plots
    if results['total_sentences'] > 0:
        plot_distribution(results['word_counts'])
        print("\nDistribution plots saved as 'sentence_length_distribution.png' and 'common_sentence_lengths.png'")