import json
import argparse
import logging
from collections import defaultdict, Counter
from pathlib import Path
import statistics
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_json_file(filepath):
    """Load and return JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {filepath}.")
        return None


def count_pos_examples(data_dict, pos_tags):
    """
    Count examples for each part of speech across all captions.

    Args:
        data_dict: Dictionary containing the hard examples
        pos_tags: List of POS tags to count

    Returns:
        Dictionary with POS counts and statistics
    """
    pos_counts = defaultdict(list)  # List of counts per caption for each POS
    total_captions = 0
    captions_with_examples = 0

    for lookup_key, examples_list in data_dict.items():
        total_captions += 1
        caption_has_examples = False

        # Count examples for each POS in this caption
        caption_pos_counts = Counter()

        for example in examples_list:
            if len(example) >= 2:  # Make sure we have both text and metadata
                metadata = str(example[1]) if len(example) > 1 else ""

                # Check which POS tags are present in this example
                for pos_tag in pos_tags:
                    if pos_tag in metadata:
                        caption_pos_counts[pos_tag] += 1
                        caption_has_examples = True

        # Add counts for this caption (0 if no examples found for a POS)
        for pos_tag in pos_tags:
            pos_counts[pos_tag].append(caption_pos_counts[pos_tag])

        if caption_has_examples:
            captions_with_examples += 1

    # Calculate statistics for each POS
    pos_stats = {}
    for pos_tag in pos_tags:
        counts = pos_counts[pos_tag]
        non_zero_counts = [c for c in counts if c > 0]

        pos_stats[pos_tag] = {
            'total_examples': sum(counts),
            'captions_with_examples': len(non_zero_counts),
            'captions_without_examples': len(counts) - len(non_zero_counts),
            'average_per_caption': statistics.mean(counts) if counts else 0,
            'average_per_caption_with_examples': statistics.mean(non_zero_counts) if non_zero_counts else 0,
            'median_per_caption': statistics.median(counts) if counts else 0,
            'std_dev': statistics.stdev(counts) if len(counts) > 1 else 0,
            'min_examples': min(counts) if counts else 0,
            'max_examples': max(counts) if counts else 0
        }

    return pos_stats, total_captions, captions_with_examples


def create_pos_comparison_plot(stats1, stats2, pos_tags, file1_name, file2_name):
    """
    Create a plotly bar chart comparing POS statistics between two files.

    Args:
        stats1: Statistics dictionary for first file
        stats2: Statistics dictionary for second file
        pos_tags: List of POS tags
        file1_name: Display name for first file
        file2_name: Display name for second file
    """
    # Extract total examples for each POS
    file1_totals = [stats1[pos]['total_examples'] for pos in pos_tags]
    file2_totals = [stats2[pos]['total_examples'] for pos in pos_tags]

    # Create the plot
    fig = go.Figure()

    # Add bars for first file
    fig.add_trace(go.Bar(
        name=file1_name,
        x=pos_tags,
        y=file1_totals,
        marker_color='lightcoral',
        text=file1_totals,
        textposition='auto',
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'POS: %{x}<br>' +
                      'Total Examples: %{y}<br>' +
                      '<extra></extra>'
    ))

    # Add bars for second file
    fig.add_trace(go.Bar(
        name=file2_name,
        x=pos_tags,
        y=file2_totals,
        marker_color='lightblue',
        text=file2_totals,
        textposition='auto',
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'POS: %{x}<br>' +
                      'Total Examples: %{y}<br>' +
                      '<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Hard Examples Generated per Part of Speech<br><sub>{file1_name} vs {file2_name}</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Part of Speech',
        yaxis_title='Number of Examples Generated',
        barmode='group',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=12),
        height=500
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def calculate_pos_statistics(file1_path, file2_path, file1_name="File 1", file2_name="File 2", show_plot=True):
    """
    Calculate and display POS statistics for two files.

    Args:
        file1_path: Path to first JSON file
        file2_path: Path to second JSON file
        file1_name: Display name for first file
        file2_name: Display name for second file
        show_plot: Whether to show the plotly visualization
    """

    # Load both files
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)

    if not data1 or not data2:
        print("Error: Could not load one or both files.")
        return

    print(f"Loaded {len(data1)} entries from {file1_name}")
    print(f"Loaded {len(data2)} entries from {file2_name}")

    # Parts of speech to analyze
    pos_tags = ["NOUN", "ADJ", "VERB", "ADV", "ADP"]

    # Calculate statistics for both files
    stats1, total_captions1, captions_with_examples1 = count_pos_examples(data1, pos_tags)
    stats2, total_captions2, captions_with_examples2 = count_pos_examples(data2, pos_tags)

    # Display results
    print("\n" + "=" * 100)
    print("POS STATISTICS COMPARISON")
    print("=" * 100)

    print(f"\n{file1_name} Overview:")
    print(f"  Total captions: {total_captions1}")
    print(f"  Captions with examples: {captions_with_examples1}")
    print(f"  Coverage: {captions_with_examples1 / total_captions1 * 100:.1f}%")

    print(f"\n{file2_name} Overview:")
    print(f"  Total captions: {total_captions2}")
    print(f"  Captions with examples: {captions_with_examples2}")
    print(f"  Coverage: {captions_with_examples2 / total_captions2 * 100:.1f}%")

    # Detailed POS statistics
    print(f"\n{'-' * 50} DETAILED POS STATISTICS {'-' * 50}")

    # Header
    print(
        f"\n{'POS':<6} {'File':<20} {'Total':<8} {'Avg/Cap':<8} {'Avg/Cap+':<9} {'Median':<7} {'StdDev':<7} {'Min':<5} {'Max':<5} {'Coverage':<9}")
    print("-" * 100)

    for pos_tag in pos_tags:
        # File 1 stats
        s1 = stats1[pos_tag]
        coverage1 = s1['captions_with_examples'] / total_captions1 * 100 if total_captions1 > 0 else 0
        print(
            f"{pos_tag:<6} {file1_name:<20} {s1['total_examples']:<8} {s1['average_per_caption']:<8.2f} {s1['average_per_caption_with_examples']:<9.2f} {s1['median_per_caption']:<7.1f} {s1['std_dev']:<7.2f} {s1['min_examples']:<5} {s1['max_examples']:<5} {coverage1:<8.1f}%")

        # File 2 stats
        s2 = stats2[pos_tag]
        coverage2 = s2['captions_with_examples'] / total_captions2 * 100 if total_captions2 > 0 else 0
        print(
            f"{'':6} {file2_name:<20} {s2['total_examples']:<8} {s2['average_per_caption']:<8.2f} {s2['average_per_caption_with_examples']:<9.2f} {s2['median_per_caption']:<7.1f} {s2['std_dev']:<7.2f} {s2['min_examples']:<5} {s2['max_examples']:<5} {coverage2:<8.1f}%")
        print()

    # Summary comparison
    print(f"\n{'-' * 40} SUMMARY COMPARISON {'-' * 40}")

    print(f"\n{'POS':<6} {'Metric':<25} {file1_name:<15} {file2_name:<15} {'Difference':<12}")
    print("-" * 80)

    for pos_tag in pos_tags:
        s1, s2 = stats1[pos_tag], stats2[pos_tag]

        print(
            f"{pos_tag:<6} {'Total Examples':<25} {s1['total_examples']:<15} {s2['total_examples']:<15} {s2['total_examples'] - s1['total_examples']:+<12}")
        print(
            f"{'':6} {'Avg per Caption':<25} {s1['average_per_caption']:<15.2f} {s2['average_per_caption']:<15.2f} {s2['average_per_caption'] - s1['average_per_caption']:+<12.2f}")
        print(
            f"{'':6} {'Coverage %':<25} {s1['captions_with_examples'] / total_captions1 * 100:<15.1f} {s2['captions_with_examples'] / total_captions2 * 100:<15.1f} {(s2['captions_with_examples'] / total_captions2 * 100) - (s1['captions_with_examples'] / total_captions1 * 100):+<12.1f}")
        print()

    # Create and show plot
    if show_plot:
        try:
            fig = create_pos_comparison_plot(stats1, stats2, pos_tags, file1_name, file2_name)
            fig.show()
            print("\n📊 Interactive plot displayed above!")

            # Option to save plot
            save_plot = input("\nWould you like to save the plot as an HTML file? (y/n): ").lower().strip()
            if save_plot == 'y':
                output_path = input("Enter output path (default: pos_comparison.html): ").strip()
                if not output_path:
                    output_path = "pos_comparison.html"
                fig.write_html(output_path)
                print(f"Plot saved to {output_path}")

        except ImportError:
            print("\n⚠️  Plotly not available. Install with: pip install plotly")
        except Exception as e:
            print(f"\n⚠️  Error creating plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Calculate POS statistics for two hard examples files')

    parser.add_argument('--file1',
                        help='Path to first JSON file (e.g., hard negatives)',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_all_pos_100.json')
    parser.add_argument('--file2',
                        help='Path to second JSON file (e.g., hard positives)',
                        default='/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_positives_all_pos_40.json')
    parser.add_argument('--name1', default='Hard Negatives',
                        help='Display name for first file (default: Hard Negatives)')
    parser.add_argument('--name2', default='Hard Positives',
                        help='Display name for second file (default: Hard Positives)')
    parser.add_argument('--show_plot', action='store_true',
                        help='Show the plotly visualization')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    calculate_pos_statistics(
        args.file1,
        args.file2,
        args.name1,
        args.name2,
        show_plot=args.show_plot
    )


if __name__ == "__main__":
    main()