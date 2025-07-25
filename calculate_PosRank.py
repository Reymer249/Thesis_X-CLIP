import numpy as np
import pickle
import warnings
import os
import json
import argparse

import torch
from tqdm import tqdm

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from get_xclip import get_xclip

warnings.filterwarnings("ignore")


model = None
device = None

def get_args(description='Fine evaluation on X-CLIP'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--save_dir",
                        type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--visual_path",  # visual features
                        type=str, required=True,
                        help="The path of visual features.")
    parser.add_argument("--visual_mask_path",  # visual mask
                        type=str, required=True,
                        help="The path of visual mask.")
    parser.add_argument("--test_model",  # path to the model
                        type=str, required=False, help="Initial model.")
    parser.add_argument("--test_id_path", type=str,
                        required=True, help="Initial model.")  # test_csv_ids_path
    parser.add_argument("--part_of_speech", type=str,
                         required=True, help="Part of speeh to test for. 'all' for all of them")  # test_csv_ids_path
    parser.add_argument("--all_captions_txt_path", type=str, help="Path to the txt file with all captions")
    parser.add_argument("--hard_negatives_folder_with_jsons_path", type=str,
                        help="Path to the folder with json files with hard negatives per part of speech. They were"
                             "provided by Chen")
    parser.add_argument("--calc_brit", action="store_true")

    args = parser.parse_args()
    return args


def txt_tokenizer(sentence, max_words=32):
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                     "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    tokenizer = ClipTokenizer()
    words = tokenizer.tokenize(sentence)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_words
    assert len(input_mask) == max_words
    assert len(segment_ids) == max_words

    pairs_text = np.array(input_ids)
    pairs_mask = np.array(input_mask)
    pairs_segment = np.array(segment_ids)

    return pairs_text, pairs_mask, pairs_segment


def smi_value(sent, batchs_video_features, batchs_video_mask):
    # pairs_text, pairs_mask, pairs_segment,
    input_ids, token_type_ids, t_attention_mask = txt_tokenizer(sent)
    input_ids = torch.Tensor(input_ids).to(device).unsqueeze(dim=0).long()
    token_type_ids = torch.Tensor(token_type_ids).to(device).unsqueeze(dim=0)
    t_attention_mask = torch.Tensor(t_attention_mask).to(device).unsqueeze(dim=0)

    model.eval()
    with torch.no_grad():
        eot_tokens, text_features = model.get_sequence_output(input_ids, t_attention_mask, token_type_ids, shaped=True)

    # text_features = text_features[0].unsqueeze(dim=0)
    # eot_tokens = eot_tokens[0].unsqueeze(dim=0)

    each_row = []
    attention_mask = None
    # for batch_video_features, batch_video_mask in zip(batchs_video_features, batchs_video_mask):

    # b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, seq_features, visual_output, input_mask, video_mask,
    #                                                              loose_type=model.loose_type)
    b1b2_logits, *_tmp = model.get_similarity_logits(eot_tokens, text_features, batchs_video_features,
                                                     attention_mask, batchs_video_mask, shaped=True,
                                                     loose_type=model.loose_type)
    b1b2_logits = b1b2_logits.cpu().detach().numpy().squeeze()  # 1*40
    # each_row.append(b1b2_logits)
    # each_row = np.concatenate(tuple(each_row), axis=-1)
    return b1b2_logits


def calculate_mrr(ranks):
    """Calculate Mean Reciprocal Rank"""
    return np.mean([1.0 / r for r in ranks]) if ranks else 0


def main():
    args = get_args()
    config = ['--do_eval', '--max_words', '32', '--max_frames', '12', '--freeze_layer_num', '0', '--slice_framepos',
              '2',
              '--loose_type', '--linear_patch', '2d', '--sim_header', 'seqTransf', '--pretrained_clip_name', 'ViT-B/32',
              '--output_dir', './', '--local_rank', '0',
              '--init_model', args.test_model]

    global model
    global device
    model = get_xclip(config)
    device = next(model.parameters()).device  # 与model保持一致

    feat_p = args.visual_path
    mask_p = args.visual_mask_path
    test_id_path = args.test_id_path

    test_video_ids = open(test_id_path).read().strip().split('\n')

    visual_feat = pickle.load(open(feat_p, 'rb'))
    visual_mask = pickle.load(open(mask_p, 'rb'))

    video_frame_feat = {}
    for video_id, f_feat in zip(test_video_ids, visual_feat):
        video_frame_feat[video_id] = f_feat

    video_frame_mask = {}
    for video_id, f_mask in zip(test_video_ids, visual_mask):
        video_frame_mask[video_id] = f_mask

    test_p = args.all_captions_txt_path
    msrvtt_s = open(test_p).read().strip().split('\n')
    raw_caps = {}
    for line in msrvtt_s:
        tt_sent_id = line.split(' ')[0]
        raw_caps[tt_sent_id] = ' '.join(line.split(' ')[1:])

    changeS_P = args.hard_negatives_folder_with_jsons_path
    if args.part_of_speech == 'all':
        test_list_neg = ['filtered_vatex1k5_adjective_RE20.json', 'filtered_vatex1k5_adverb_RE20.json',
                    'filtered_vatex1k5_noun_RE20.json', 'filtered_vatex1k5_preposition_RE20.json',
                    'filtered_vatex1k5_verb_RE20.json']
    else:
        try:
            test_list_neg = [f'filtered_vatex1k5_{args.part_of_speech}_RE20.json']
        except:
            raise ValueError('Invalid part of speech specified. Choose from: all, adjective, adverb, noun, preposition, verb.')

    if args.calc_brit:
        if args.part_of_speech == 'all':
            test_list_pos = ['filtered_vatex1k5_pos_adjective_RE20.json', 'filtered_vatex1k5_pos_adverb_RE20.json',
                             'filtered_vatex1k5_pos_noun_RE20.json', 'filtered_vatex1k5_pos_preposition_RE20.json',
                             'filtered_vatex1k5_pos_verb_RE20.json']
        else:
            try:
                test_list_pos = [f'filtered_vatex1k5_pos_{args.part_of_speech}_RE20.json']
            except:
                raise ValueError(
                    'Invalid part of speech specified. Choose from: all, adjective, adverb, noun, preposition, verb.')

    save_p = args.save_dir
    if not os.path.exists(save_p):
        os.mkdir(save_p)

    for position, neg in enumerate(test_list_neg):
        changejs_neg = json.load(open(os.path.join(changeS_P, neg)))
        if args.calc_brit:
            changejs_pos = json.load(open(os.path.join(changeS_P, test_list_pos[position])))
        testname = neg.split('.json')[0]

        rank_GT = []  # append rank of gt sentence
        res_list = {}
        brit_counter = 0
        brit_num_comparisons = 0

        for sent_id in tqdm(list(changejs_neg.keys())):

            video_id = sent_id.split('#')[0]
            video_feat = torch.tensor(np.array(video_frame_feat[video_id].cpu())).to(device)#.unsqueeze(dim=0)
            video_mask = torch.tensor(np.array(video_frame_mask[video_id].cpu())).to(device)#.unsqueeze(dim=0)

            # r_sent_id = sent_id.replace('#', '#enc#')
            raw_sent = raw_caps[sent_id]  # use the encoded sentence

            chage_s = list(changejs_neg[sent_id].values())
            temp_test_sent = [raw_sent] + chage_s
            if args.calc_brit:
                try:
                    change_s_pos = list(changejs_pos[sent_id].values())
                    if len(change_s_pos) != len(chage_s):  # we don't have as many hard positives as negatives
                        continue
                except KeyError:  # if we could not generate hard positive for the given hard negative sentence
                    continue
                change_s_pos = [None] + change_s_pos
            temp_sim_res = []
            for number, S in enumerate(temp_test_sent):
                temp_neg_sim = smi_value(S, video_feat, video_mask)
                temp_sim_res.append(temp_neg_sim)
                if args.calc_brit and number > 0:  # not the firs one, as it is the original one
                    temp_pos_sim = smi_value(change_s_pos[number], video_feat, video_mask)
                    brit_num_comparisons += 1
                    if temp_sim_res[0] > temp_neg_sim > temp_pos_sim or temp_pos_sim > temp_neg_sim > temp_sim_res[0]:
                        brit_counter += 1
            temp_sim_res = np.array(temp_sim_res).squeeze()
            temp_sort_res = np.argsort(temp_sim_res)  # [-1] is the index of largest one

            temp_gt_rank = list(temp_sort_res)[::-1].index(0) + 1
            res_list[sent_id] = [int(x) for x in list(temp_sort_res)[::-1]]
            rank_GT.append(int(temp_gt_rank))

        wr = open(os.path.join(save_p, '{}_dump_antonym_v2tres_MRR.txt'.format(testname)), 'w')
        print(len(rank_GT))
        meanR_rank_GT = sum(rank_GT) / len(rank_GT)
        mrr = calculate_mrr(rank_GT)
        print('Mean Reciprocal Rank of  {} is:'.format(testname), mrr)
        wr.write('meanR of change {} is:{}\n'.format(testname, mrr))
        avg_rank_GT = rank_GT.count(1) / len(rank_GT)
        print('GT @ R1 count {} is:'.format(testname), rank_GT.count(1))
        wr.write('GT @ R1 count {} is:{}\n'.format(testname, rank_GT.count(1)))
        if args.calc_brit:
            brit_result_str = f"Brittleness: {brit_counter/brit_num_comparisons} ({brit_counter}/{brit_num_comparisons})"
            print(brit_result_str)
            wr.write(brit_result_str)
        wr.close()

        # with open(os.path.join(save_p, '{}_dump_antonym_v2tres.json'.format(testname)), 'w') as json_file:
        #     json_file.write(json.dumps(res_list))


if __name__ == "__main__":
    main()
