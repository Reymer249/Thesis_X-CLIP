# ViT-B/16
job_name="xclip_vatex_vit16"
DATA_PATH="/home/s3705609/data1"
python -m torch.distributed.run --nproc_per_node=2 --master_port=$(($RANDOM + 10000))\
    main_xclip.py --do_train --num_thread_reader=2 \
    --epochs=5 --batch_size=16 --n_display=50 \
    --data_path ${DATA_PATH}/VATEX \
    --features_path ${DATA_PATH}/VATEX/clips \
    --output_dir ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype vatex \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --n_gpu 2 --fp16 --fp16_opt_level O3 \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a /home/s3705609/data1/log/${job_name} \
