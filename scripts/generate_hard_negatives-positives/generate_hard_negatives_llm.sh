python /home/s3705609/thesis_code/helpers_scripts/hard_negatives-positives_generation/generate_hard_positives-negatives_llm.py \
  --captions_file /vol/home/s3705609/Desktop/data_vatex/splits_txt/captions_avail_formatted.json \
  --video_ids_file /vol/home/s3705609/Desktop/data_vatex/splits_txt/test.txt \
  --num_sentences 20 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --do_neg_gen