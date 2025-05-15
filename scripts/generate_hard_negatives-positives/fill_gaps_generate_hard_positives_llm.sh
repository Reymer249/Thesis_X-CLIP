python /home/s3705609/thesis_code/helpers_scripts/hard_negatives-positives_generation/fill_gaps_generate_hard_positives-negatives_llm.py \
  --captions_file /data/s3705609/VATEX/splits_txt/captions_train_avail_020.json \
  --generated_file /data/s3705609/VATEX/splits_txt/hard_positives_llm.json \
  --num_sentences 20 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --batch_size 128