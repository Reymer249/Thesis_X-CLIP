python /vol/home/s3705609/Desktop/thesis_code/helpers_scripts/hard_negatives-positives_generation/fill_gaps_generate_hard_positives-negatives_llm.py \
  --captions_file /vol/home/s3705609/Desktop/data_vatex/splits_txt/captions_avail_train_020.json \
  --generated_file /vol/home/s3705609/Desktop/data_vatex/llm_sentences/hard_negatives_final.json \
  --num_sentences 20 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --batch_size 16 \
  --gen_hard_neg