import json
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

from constants import *

type = "Negative"
set_size = 100
file_path = f"/vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_{type.lower()}s_all_pos_{set_size}.json"
# Load the JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Count the number of sentences per video ID
sentences_per_video = {video_id: len(sentences) for video_id, sentences in data.items()}

# Create a counter to get the distribution
counts = Counter(sentences_per_video.values())

# Convert to dataframe for easier plotting
df = pd.DataFrame(sorted(counts.items()), columns=[f'Number of Hard {type} Sentences', 'Frequency'])

# Create the plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df[f'Number of Hard {type} Sentences'],
    y=df['Frequency'],
    marker_color='skyblue',
    marker_line_color='black',
    marker_line_width=1.5,
    opacity=0.7
))

coef = 0.66

# Update layout
fig.update_layout(
    title=f'Distribution of Hard {type} Sentences per Caption (max length is {set_size})',
    xaxis_title='Number of Sentences',
    yaxis_title='Frequency (Number of Captions)',
    bargap=0,
    font=dict(
            family="Arial, sans-serif",
            size=18,  # Larger base font size
            color="#333"
        ),
    height=1080,
    width=1920,
)

fig.update_xaxes(
    title_font=dict(size=int(xaxis_title_size*coef), family="Arial, sans-serif"),
    tickfont=dict(size=int(xaxis_tickfont_size*coef))
)
fig.update_yaxes(
    title_font=dict(size=int(yaxis_title_size*coef), family="Arial, sans-serif"),
    tickfont=dict(size=int(yaxis_tickfont_size*coef))
)

fig.update_layout(
    title={
        "font": {"size": int(title_font*coef)},
    }
)

# Show statistics
total_videos = len(sentences_per_video)
avg_sentences = sum(sentences_per_video.values()) / total_videos
max_sentences = max(sentences_per_video.values())
min_sentences = min(sentences_per_video.values())

print(f"Total number of videos: {total_videos}")
print(f"Average sentences per video: {avg_sentences:.2f}")
print(f"Maximum sentences for a video: {max_sentences}")
print(f"Minimum sentences for a video: {min_sentences}")

# Find videos with most and least sentences
videos_with_most = [vid for vid, count in sentences_per_video.items() if count == max_sentences]
videos_with_least = [vid for vid, count in sentences_per_video.items() if count == min_sentences]

print(f"\nVideos with most sentences ({max_sentences}):")
for vid in videos_with_most[:5]:  # Show first 5 examples
    print(f"- {vid}")
if len(videos_with_most) > 5:
    print(f"  ...and {len(videos_with_most) - 5} more")

print(f"\nVideos with least sentences ({min_sentences}):")
for vid in videos_with_least[:5]:  # Show first 5 examples
    print(f"- {vid}")
if len(videos_with_least) > 5:
    print(f"  ...and {len(videos_with_least) - 5} more")

# Show plot
fig.write_image(f"disrt_{type.lower()}_sentences_set_{set_size}.svg")

# Save plot if needed
# fig.write_html("video_sentence_distribution.html")