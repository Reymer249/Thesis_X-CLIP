import json
import random
import os
from typing import List, Tuple, Dict


class HardNegativeOrPositiveSampler:
    """
    Class to load and sample hard negative sentences from a JSON file.
    """

    def __init__(self, json_file_path: str):
        """
        Initialize the sampler with the path to the JSON file containing positive or negative samples.

        Args:
            json_file_path: Path to the JSON file with positive or negative samples
        """
        self.file_path = json_file_path
        self.changed_sentences_dict = self._load_positives_or_negatives()

    def _load_positives_or_negatives(self) -> Dict[str, List]:
        """
        Load positive/negative samples from the JSON file.

        Returns:
            Dict mapping video_id#sentence_num to list of [hard_neg, pos] pairs
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Positive/Negative samples file not found: {self.file_path}")

        with open(self.file_path, 'r') as f:
            changed_sentences_dict = json.load(f)

        print(f"Loaded {len(changed_sentences_dict)} entries from positive/negative samples file")
        return changed_sentences_dict

    def get_neg_word_level_sentences(self, caption: str, video_id: str = None,
                                     sentence_num: int = 0, change_num: int = 15) -> Tuple[List[str], List[int]]:
        """
        Get hard positive/negative sentences for a caption from the loaded JSON file.

        Args:
            caption: Original caption text
            video_id: Video ID associated with the caption
            sentence_num: Sentence number for the video
            change_num: Number of hard positives/negatives to sample

        Returns:
            Tuple of:
                - List of sentences (original + sampled positives/negatives)
                - List of positions that were changed (dummy list for compatibility)
        """
        # Create the key to lookup in the JSON file
        key = f"{video_id}#{sentence_num}" if video_id is not None else None

        neg_sentences = []
        change_positions = []

        # First element is always the original caption
        neg_sentences.append(caption)
        change_positions.append(None)  # No change for the original

        # If we have the key in our dictionary, sample from there
        assert key is not None and key in self.changed_sentences_dict

        pairs = self.changed_sentences_dict[key]

        # Sample change_num-1 items as we SHOULD have more hard positives/negatives than change_num
        sampled_pairs = random.sample(pairs, change_num - 1)

        for neg_pair in sampled_pairs:
            hard_neg, pos = neg_pair
            neg_sentences.append(hard_neg)
            change_positions.append(pos)

        return neg_sentences, change_positions
