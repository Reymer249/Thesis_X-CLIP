MODEL_NAME=$1 # name of the folder where the weights of the models are located
DATA_FOLDER="/vol/home/s3705609/Desktop/data_vatex"

python calculate_PosRank.py \
  --save_dir ${DATA_FOLDER}/x-clip_checkpoints/${MODEL_NAME}/results \
  --visual_path ${DATA_FOLDER}/x-clip_checkpoints/${MODEL_NAME}/vatex_val_video_features/batch_visual_output_list.pkl \
  --visual_mask_path ${DATA_FOLDER}/x-clip_checkpoints/${MODEL_NAME}/vatex_val_video_features/batch_video_mask_list.pkl \
  --test_model ${DATA_FOLDER}/x-clip_checkpoints/${MODEL_NAME}/pytorch_model.bin.4 \
  --test_id_path ${DATA_FOLDER}/splits_txt/vatex_val_avail.txt \
  --part_of_speech $2 \
  --all_captions_txt_path /vol/home/s3705609/Desktop/data_vatex/splits_txt/captions.txt \
  --hard_negatives_folder_with_jsons_path /vol/home/s3705609/Desktop/data_vatex/splits_txt/hard_negatives_chen_provided/Chen_filtered

# For part_of_speech options are [all, adjective, adverb, noun, preposition, verb]