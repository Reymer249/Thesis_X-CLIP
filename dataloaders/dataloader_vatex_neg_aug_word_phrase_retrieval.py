from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor
from dotenv import load_dotenv

import json
from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import wordnet
import random

# 下载WordNet语料库
nltk.download('wordnet')
import spacy
from spacy import displacy
import textacy

cls = spacy.util.get_lang_class('en')
stop_words = cls.Defaults.stop_words
nlp = spacy.load("en_core_web_sm")

from dataloaders.hard_negatives_sampler import HardNegativeOrPositiveSampler


class VATEX_TrainDataLoader(Dataset):
    """UVO dataset loader."""

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            output_dir,
            train_path_from_data_folder,
            test_path_from_data_folder,
            val_path_from_data_folder,
            captions_path_from_data_folder,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            do_neg_aug=False,
            hard_negatives_json_path=None,
            neg_aug_num_sentences=16,
            do_pos_aug=False,
            hard_positives_json_path=None,
            pos_aug_num_sentences=16,
    ):
        self.output_dir = output_dir
        self.temp_wr = open('{}/neg_train.txt'.format(self.output_dir), 'a')
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.do_neg_aug = do_neg_aug
        self.neg_aug_num_sentences = neg_aug_num_sentences
        if self.do_neg_aug:
            self.HardNegSampler = HardNegativeOrPositiveSampler(json_file_path=hard_negatives_json_path)

        self.do_pos_aug = do_pos_aug
        self.pos_aug_num_sentences = pos_aug_num_sentences
        if self.do_pos_aug:
            self.HardPosSampler = HardNegativeOrPositiveSampler(json_file_path=hard_positives_json_path)

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, train_path_from_data_folder)
        video_id_path_dict["val"] = os.path.join(self.data_path, val_path_from_data_folder)
        video_id_path_dict["test"] = os.path.join(self.data_path, test_path_from_data_folder)
        caption_file = os.path.join(self.data_path, captions_path_from_data_folder)

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        captions = json.load(open(caption_file))

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for caption_num, cap_txt in enumerate(captions[video_id]):
                self.sentences_dict[len(self.sentences_dict)] = (video_id, caption_num, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))
        print("Total Paire: {} {}".format(self.subset, len(self.sentences_dict)))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True  # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {} {}".format(self.subset, len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_text_word(self, video_id, caption, caption_number, change_num_neg=16, change_num_pos=16):
        k = 1  # batch size for the neg gen
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        # 产生change_num个 hard negative samples
        # neg_word_level_sents,change_word_pos = get_neg_word_level_sent_fun(caption,change_num=change_num)#generated K sentences but only use K-1 as hard negative samples
        ## 输出 word-level and phrase-level
        if self.do_neg_aug:
            word_text_neg = np.zeros((k, change_num_neg, self.max_words), dtype=np.int64)
            word_mask_neg = np.zeros((k, change_num_neg, self.max_words), dtype=np.int64)
            word_segment_neg = np.zeros((k, change_num_neg, self.max_words), dtype=np.int64)
            word_neg_sents, word_neg_change_pos = self.HardNegSampler.get_neg_word_level_sentences(
                video_id=video_id,
                caption=caption,
                sentence_num=caption_number,
                change_num=change_num_neg
            )
            # generated K sentences but only use K-1 as hard negative samples

        if self.do_pos_aug:
            word_text_pos = np.zeros((k, change_num_pos, self.max_words), dtype=np.int64)
            word_mask_pos = np.zeros((k, change_num_pos, self.max_words), dtype=np.int64)
            word_segment_pos = np.zeros((k, change_num_pos, self.max_words), dtype=np.int64)
            word_pos_sents, word_pos_change_pos = self.HardPosSampler.get_neg_word_level_sentences(
                video_id=video_id,
                caption=caption,
                sentence_num=caption_number,
                change_num=change_num_pos
            )
            # generated K sentences but only use K-1 as hard positive samples (same logic as Chen)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            # negative
            if self.do_neg_aug:
                for j, t_pos_sent in enumerate(word_neg_sents):
                    words = self.tokenizer.tokenize(t_pos_sent)
                    words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                    total_length_with_CLS = self.max_words - 1
                    if len(words) > total_length_with_CLS:
                        words = words[:total_length_with_CLS]
                    words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                    input_ids_aug = self.tokenizer.convert_tokens_to_ids(words)
                    input_mask_aug = [1] * len(input_ids_aug)
                    segment_ids_aug = [0] * len(input_ids_aug)

                    while len(input_ids_aug) < self.max_words:
                        input_ids_aug.append(0)
                        input_mask_aug.append(0)
                        segment_ids_aug.append(0)
                    assert len(input_ids_aug) == self.max_words
                    assert len(input_mask_aug) == self.max_words
                    assert len(segment_ids_aug) == self.max_words

                    if j == i:  # i和j 因为i始终为0 所以，每一个list中，第一个句子为original sentence
                        word_text_neg[i][j] = np.array(input_ids)
                        word_mask_neg[i][j] = np.array(input_mask)
                        word_segment_neg[i][j] = np.array(segment_ids)
                    else:
                        word_text_neg[i][j] = np.array(input_ids_aug)
                        word_mask_neg[i][j] = np.array(input_mask_aug)
                        word_segment_neg[i][j] = np.array(segment_ids_aug)

            # positive
            if self.do_pos_aug:
                for j, t_pos_sent in enumerate(word_pos_sents):
                    words = self.tokenizer.tokenize(t_pos_sent)
                    words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                    total_length_with_CLS = self.max_words - 1
                    if len(words) > total_length_with_CLS:
                        words = words[:total_length_with_CLS]
                    words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                    input_ids_aug = self.tokenizer.convert_tokens_to_ids(words)
                    input_mask_aug = [1] * len(input_ids_aug)
                    segment_ids_aug = [0] * len(input_ids_aug)

                    while len(input_ids_aug) < self.max_words:
                        input_ids_aug.append(0)
                        input_mask_aug.append(0)
                        segment_ids_aug.append(0)
                    assert len(input_ids_aug) == self.max_words
                    assert len(input_mask_aug) == self.max_words
                    assert len(segment_ids_aug) == self.max_words

                    if j == i:  # i和j 因为i始终为0 所以，每一个list中，第一个句子为original sentence
                        word_text_pos[i][j] = np.array(input_ids)
                        word_mask_pos[i][j] = np.array(input_mask)
                        word_segment_pos[i][j] = np.array(segment_ids)
                    else:
                        word_text_pos[i][j] = np.array(input_ids_aug)
                        word_mask_pos[i][j] = np.array(input_mask_aug)
                        word_segment_pos[i][j] = np.array(segment_ids_aug)

        if self.do_neg_aug and self.do_pos_aug:
            return pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_neg, word_mask_neg,\
                word_segment_neg, word_text_pos, word_mask_pos, word_segment_pos
        elif self.do_neg_aug:
            return pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_neg, word_mask_neg,\
                word_segment_neg
        elif self.do_pos_aug:
            return pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_pos, word_mask_pos,\
                word_segment_pos
        else:
            raise Exception("Neither pos_aug nor neg_aug was specified, but _gen_hard_sentences function was called")

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption_number, caption = self.sentences_dict[idx]

        if self.do_neg_aug and self.do_pos_aug:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_neg, word_mask_neg, word_segment_neg, \
                word_text_pos, word_mask_pos, word_segment_pos = self._get_text_word(
                video_id, caption, caption_number,
                change_num_neg=self.neg_aug_num_sentences,
                change_num_pos=self.pos_aug_num_sentences
            )
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, word_text_neg, word_mask_neg,\
                word_segment_neg, word_text_pos, word_mask_pos, word_segment_pos
        elif self.do_neg_aug:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_neg, word_mask_neg, word_segment_neg = \
                self._get_text_word(
                video_id, caption, caption_number,
                    change_num_neg=self.neg_aug_num_sentences
            )
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, word_text_neg, word_mask_neg, \
                word_segment_neg
        elif self.do_pos_aug:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_pos, word_mask_pos, word_segment_pos = \
                self._get_text_word(
                    video_id, caption, caption_number,
                    change_num_pos=self.pos_aug_num_sentences
                )
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, word_text_pos, word_mask_pos, \
                word_segment_pos
        else:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
