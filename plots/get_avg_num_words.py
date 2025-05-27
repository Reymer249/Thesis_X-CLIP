import json
import statistics
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
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
    # Create a Plotly subplot figure
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Distribution of Words per Sentence',
                                        'Box Plot of Words per Sentence'),
                        vertical_spacing=0.15,  # Increased spacing for better readability
                        specs=[[{"type": "histogram"}],
                               [{"type": "box"}]])

    # 1. Histogram
    fig.add_trace(
        go.Histogram(
            x=word_counts,
            nbinsx=max(word_counts) - min(word_counts) + 1,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1.5,  # Slightly thicker lines
            opacity=0.7,
            name='Frequency'
        ),
        row=1, col=1
    )

    # 3. Box Plot
    fig.add_trace(
        go.Box(
            x=word_counts,
            marker_color='lightseagreen',
            name='Word Counts',
            boxmean=True,  # adds a marker for the mean
            line=dict(width=2)  # Thicker box lines
        ),
        row=2, col=1
    )

    # Update layout with larger text sizes
    fig.update_layout(
        height=1240,
        width=1080,
        showlegend=False,
        font=dict(
            family="Arial, sans-serif",
            size=18,  # Larger base font size
            color="#333"
        ),
    )

    # Update subplot titles with larger font
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 24

    # Update x and y axis labels with larger font sizes
    fig.update_xaxes(
        title_text="Number of Words",
        title_font=dict(size=20, family="Arial, sans-serif"),
        tickfont=dict(size=16),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Frequency",
        title_font=dict(size=20, family="Arial, sans-serif"),
        tickfont=dict(size=16),
        row=1, col=1
    )

    fig.update_xaxes(
        title_text="Number of Words",
        title_font=dict(size=20, family="Arial, sans-serif"),
        tickfont=dict(size=16),
        row=2, col=1
    )
    fig.update_yaxes(
        tickfont=dict(size=16),
        row=2, col=1
    )

    # Save as HTML file for interactivity
    fig.write_html('sentence_length_distribution.html')

    # Create a frequency distribution for the most common sentence lengths
    counter = Counter(word_counts)
    most_common = counter.most_common(10)

    labels = [str(x[0]) for x in most_common]
    values = [x[1] for x in most_common]

    # Bar chart for most common sentence lengths
    bar_fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1.5
        )
    ])

    bar_fig.update_layout(
        title={
            'text': 'Most Common Sentence Lengths',
            'font': {'size': 28, 'family': 'Arial, sans-serif', 'color': '#333'},
            'y': 0.95
        },
        xaxis_title={
            'text': 'Number of Words',
            'font': {'size': 22, 'family': 'Arial, sans-serif'}
        },
        yaxis_title={
            'text': 'Frequency',
            'font': {'size': 22, 'family': 'Arial, sans-serif'}
        },
        height=800,
        width=1200,  # Larger size for better visibility
        font=dict(
            family="Arial, sans-serif",
            size=18,
            color="#333"
        ),
        xaxis=dict(
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            tickfont=dict(size=16)
        ),
        margin=dict(l=80, r=80, t=100, b=80)  # Increased margins for better spacing
    )

    # Save as HTML file
    bar_fig.write_html('common_sentence_lengths.html')


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
        print("\nDistribution plots saved as 'sentence_length_distribution.html' and 'common_sentence_lengths.html'")