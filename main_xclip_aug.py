from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import wandb  # Import wandb
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_xclip_aug import XCLIP
from modules.optimization import BertAdam
from dotenv import load_dotenv

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")

global logger


def init_distributed_mode():
    """Initialize distributed mode with proper device assignment."""
    # Check if we're running via torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Print debugging info
        print(f"[Process {os.getpid()}] rank = {rank}, world_size = {world_size}, local_rank = {local_rank}")

        # Set the device BEFORE initializing process group
        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        print(f"[Process {os.getpid()}] Initialized process group: rank={rank}, world_size={world_size}")

        # Synchronize all processes
        torch.distributed.barrier()

        return local_rank
    else:
        # Fallback for single GPU or non-distributed training
        print("No distributed environment variables found. Running in non-distributed mode.")
        return 0  # Default to first GPU


def get_args(description='X-CLIP on hard negatives augmented Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the val set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument('--loss_func', type=str, default="maxcol_word",
                        choices=["maxcol_word"],
                        help="choice a loss function.")
    parser.add_argument('--rank_margin', type=float, default=0.2, help='rank margin for hierachicalRank loss')
    parser.add_argument('--do_neg_aug', action="store_true", default=False,
                        help='Whether to do negative augmentation (hard negatives) or not')
    parser.add_argument('--neg_aug_num_sentences', type=int, default=0,
                        help='Number of hard negatives to generate for every caption. That is the total number of'
                             'hard negatives (not per part-of-speech group). We randomly select which part fo speech to'
                             'change.')
    parser.add_argument('--do_pos_aug', action="store_true", default=False,
                        help='Whether to do positive augmentation (hard positives) or not')
    parser.add_argument('--pos_aug_num_sentences', type=int, default=0,
                        help='Number of hard positive to generate for every caption. That is the total number of'
                             'hard positives (not per part-of-speech group). We randomly select which part fo speech to'
                             'change.')

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--train_path_from_data_folder", type=str, required=True,
                        help="The path to the txt file with video ids for training (video id per line) from the data folder path")
    parser.add_argument("--val_path_from_data_folder", type=str, required=True,
                        help="The path to the txt file with video ids for validation (video id per line) from the data folder path")
    parser.add_argument("--test_path_from_data_folder", type=str, required=True,
                        help="The path to the txt file with video ids for testing (video id per line) from the data folder path")
    parser.add_argument("--captions_path_from_data_folder", type=str, required=True,
                        help="The path to the json file with video captions from the data folder path")
    parser.add_argument("--hard_negatives_json_path", type=str, required=True,
                        help="The path to the json file with hard negative sentences")

    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="x-clip", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated list of tags for the run")
    parser.add_argument("--wandb_watch", action='store_true', help="Whether to watch model parameters and gradients")

    # Remove the conflicting local_rank argument
    # parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

    args = parser.parse_args()

    # Get local_rank from environment variable instead
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def init_wandb(args, model_config=None):
    """Initialize Weights & Biases logging if enabled."""
    if args.use_wandb and args.local_rank == 0:  # Only log on main process
        # Setup wandb tags if provided
        tags = None
        if args.wandb_tags:
            tags = [tag.strip() for tag in args.wandb_tags.split(',')]

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=tags,
            config={
                # Training hyperparameters
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size * args.gradient_accumulation_steps,
                "warmup_proportion": args.warmup_proportion,
                "weight_decay": 0.2,  # from prep_optimizer

                # Model configuration
                "datatype": args.datatype,
                "pretrained_clip_name": args.pretrained_clip_name,
                "sim_header": args.sim_header,
                "freeze_layer_num": args.freeze_layer_num,
                "text_num_hidden_layers": args.text_num_hidden_layers,
                "visual_num_hidden_layers": args.visual_num_hidden_layers,
                "cross_num_hidden_layers": args.cross_num_hidden_layers,
                "linear_patch": args.linear_patch,

                # Other settings
                "fp16": args.fp16,
                "seed": args.seed,
            }
        )

        # Add any additional model config if provided
        if model_config is not None:
            wandb.config.update(model_config)

        logger.info("Weights & Biases logging enabled")


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    # Check which GPUs are visible
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

    # Set device explicitly based on local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Get total number of available GPUs
    total_gpus = torch.cuda.device_count()
    logger.info(f"Total available GPUs: {total_gpus}")

    # In distributed training, each process uses one GPU
    n_gpu = 1

    logger.info(f"Process {os.getpid()} using device: {device}, local_rank: {local_rank}")
    logger.info(f"Current device index: {torch.cuda.current_device()}")
    logger.info(f"Device properties: {torch.cuda.get_device_properties(local_rank)}")

    # Store total number of GPUs in args for batch size calculations
    args.n_gpu = total_gpus

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            f"Invalid batch_size/batch_size_val and n_gpu parameter: {args.batch_size}%{args.n_gpu} and {args.batch_size_val}%{args.n_gpu}, should be == 0")

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    # Create DDP model with explicit device_ids
    # This is the key part to ensure proper GPU assignment
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],  # Use local_rank as device_id
        output_device=local_rank,
        find_unused_parameters=True
    )

    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
    }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                      task_config=args)

        model.to(device)
    else:
        model = None
    return model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, input_ids_aug, input_mask_aug, segment_ids_aug = batch
        loss = model(input_ids, segment_ids, input_mask, video, input_ids_aug, segment_ids_aug, input_mask_aug, video_mask=video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1

            # Log training progress
            if global_step % log_step == 0 and local_rank == 0:
                # Get current learning rate
                current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                lr_str = "-".join([str('%.9f' % itm) for itm in sorted(list(set(current_lr)))])

                # Calculate and log metrics
                time_per_step = (time.time() - start_time) / (log_step * args.gradient_accumulation_steps)

                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f",
                            epoch + 1, args.epochs, step + 1, len(train_dataloader),
                            lr_str, float(loss), time_per_step)

                # Log to wandb if enabled
                if args.use_wandb and local_rank == 0:
                    # Log scale parameter as well, which is important for CLIP models
                    logit_scale = 0
                    if hasattr(model, 'module') and hasattr(model.module, 'clip') and hasattr(model.module.clip,
                                                                                              'logit_scale'):
                        logit_scale = model.module.clip.logit_scale.exp().item()
                    elif hasattr(model, 'clip') and hasattr(model.clip, 'logit_scale'):
                        logit_scale = model.clip.logit_scale.exp().item()

                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": float(loss),
                        "train/learning_rate": float(current_lr[0]),  # Log first LR
                        "train/logit_scale": logit_scale,
                        "train/global_step": global_step,
                        "train/time_per_step": time_per_step,
                        "train/progress": step / len(train_dataloader)
                    }, step=global_step)

                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list,
                       batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        seq_features = batch_seq_features_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, seq_features, visual_output, input_mask,
                                                             video_mask,
                                                             loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu, split="val"):
    """Evaluate the model on validation or test set"""
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_seq_features_list = []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):  # Maybe something went wrong here!!!
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output, seq_features = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                (sequence_output, seq_features), visual_output = model.get_sequence_visual_output(input_ids,
                                                                                                  segment_ids,
                                                                                                  input_mask, video,
                                                                                                  video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list,
                                        batch_seq_features_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
        format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    # Log metrics to wandb if enabled (only on main process)
    if args.use_wandb and args.local_rank == 0:
        metrics_dict = {
            f"{split}/t2v_r1": tv_metrics['R1'],
            f"{split}/t2v_r5": tv_metrics['R5'],
            f"{split}/t2v_r10": tv_metrics['R10'],
            f"{split}/t2v_mean_r": tv_metrics['MeanR'],
            f"{split}/t2v_median_r": tv_metrics['MR'],
            f"{split}/v2t_r1": vt_metrics['R1'],
            f"{split}/v2t_r5": vt_metrics['R5'],
            f"{split}/v2t_r10": vt_metrics['R10'],
            f"{split}/v2t_mean_r": vt_metrics['MeanR'],
            f"{split}/v2t_median_r": vt_metrics['MR'],
        }
        wandb.log(metrics_dict)

    R1 = tv_metrics['R1']
    return R1


def main(args):
    global logger
    local_rank = init_distributed_mode()
    args.local_rank = local_rank
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    # Initialize wandb if enabled (only on the main process)
    if args.use_wandb and args.local_rank == 0:
        wandb.login(key=WANDB_KEY)
        init_wandb(args)
        logger.info("Weights & Biases logging initialized")

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    # Log model architecture if wandb is enabled
    if local_rank == 0 and args.wandb_watch:
        if hasattr(model, "module"):
            wandb.watch(model.module, log="all", log_freq=args.n_display)
        else:
            wandb.watch(model, log="all", log_freq=args.n_display)

    # Log model architecture details if available
    if args.use_wandb and args.local_rank == 0 and hasattr(model, "config"):
        wandb.config.update({"model_config": model.config})

    ## ####################################
    # freeze testing
    ## ####################################
    frozen_params = 0
    trainable_params = 0

    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                trainable_params += param.numel()
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    trainable_params += param.numel()
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                trainable_params += param.numel()
                continue
            else:
                # parameters which < freeze_layer_num will be frozen
                param.requires_grad = False
                frozen_params += param.numel()

    # Count all trainable parameters for logging
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()

    # Log parameter counts to wandb
    if args.use_wandb and args.local_rank == 0:
        wandb.config.update({
            "frozen_params": frozen_params,
            "trainable_params": trainable_params,
            "total_params": frozen_params + trainable_params,
            "trainable_percentage": trainable_params / (frozen_params + trainable_params) * 100
        })
        logger.info(f"Model has {trainable_params:,} trainable parameters and {frozen_params:,} frozen parameters")

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results_batch16 if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

        # Log dataset info to wandb
        if args.use_wandb:
            wandb.config.update({
                "train_examples": train_length if 'train_length' in locals() else 0,
                "val_examples": val_length,
                "test_examples": test_length,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "val_batch_size": args.batch_size_val
            })

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        start_time = time.time()
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                     args.local_rank, coef_lr=coef_lr)

        # Update wandb config with optimizer details
        if args.use_wandb and args.local_rank == 0:
            # Add optimizer and scheduler info to wandb config
            optimizer_config = {
                "optimizer": optimizer.__class__.__name__,
                "lr_scheduler": "warmup_cosine",
                "warmup_proportion": args.warmup_proportion,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "total_train_steps": num_train_optimization_steps,
            }
            wandb.config.update(optimizer_config)

            # Set up wandb to watch model weights and gradients
            if args.wandb_watch:
                if hasattr(model, "module"):
                    wandb.watch(model.module, log="all", log_freq=args.n_display)
                else:
                    wandb.watch(model, log="all", log_freq=args.n_display)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        best_epoch = -1

        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            # Load model weights
            # Load model weights
            model_checkpoint = torch.load(args.resume_model, map_location='cpu')

            # Check if model_checkpoint is already a state dict or needs to be accessed as state_dict
            if not isinstance(model_checkpoint, dict) or 'state_dict' in model_checkpoint:
                model_checkpoint = model_checkpoint.get('state_dict', model_checkpoint)

            # Handle the case where the saved model wasn't wrapped in DDP but current one is
            if hasattr(model, 'module'):
                # Option 1: Add 'module.' prefix to keys if missing
                if not all(k.startswith('module.') for k in model_checkpoint.keys()):
                    model_checkpoint = {'module.' + k: v for k, v in model_checkpoint.items()}
            else:
                # Option 2: Remove 'module.' prefix if present
                if all(k.startswith('module.') for k in model_checkpoint.keys()):
                    model_checkpoint = {k.replace('module.', ''): v for k, v in model_checkpoint.keys()}

            # Load with strict=False to ignore missing or unexpected keys
            model.load_state_dict(model_checkpoint)

            # Load optimizer state from the corresponding optimizer file
            # Convert pytorch_model.bin.2 path to pytorch_opt.bin.2
            optimizer_path = args.resume_model.replace("pytorch_model", "pytorch_opt")

            if os.path.exists(optimizer_path):
                optimizer_checkpoint = torch.load(optimizer_path, map_location='cpu')
                optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
                resumed_epoch = optimizer_checkpoint['epoch'] + 1
                resumed_loss = optimizer_checkpoint['loss']
                print(f"Resumed training from epoch {resumed_epoch}")
            else:
                raise Exception("No optimizer file found (pytorch_opt.bla-bla-bla)")

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)

            epoch_time = time.time() - epoch_start_time

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f, Time: %f s",
                            epoch + 1, args.epochs, tr_loss, epoch_time)

                # Log epoch summary to wandb
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/epoch_loss": tr_loss,
                        "train/epoch_time": epoch_time,
                        "train/epoch_time_hrs": epoch_time / 3600,
                        "train/epoch_completed": (epoch + 1) / args.epochs,
                    })

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset for selecting best model.
                logger.info("Eval on val dataset")
                val_start_time = time.time()
                R1 = eval_epoch(args, model, val_dataloader, device, n_gpu, split="val")
                val_time = time.time() - val_start_time

                if args.use_wandb:
                    wandb.log({
                        "val/evaluation_time": val_time,
                        "epoch": epoch + 1
                    })

                if best_score <= R1:
                    best_score = R1
                    best_epoch = epoch + 1
                    best_output_model_file = output_model_file

                    # Log best model to wandb
                    if args.use_wandb:
                        wandb.run.summary["best_val_r1"] = R1
                        wandb.run.summary["best_model_epoch"] = best_epoch
                        wandb.run.summary["best_model_path"] = best_output_model_file

                        # Optional: save the best model to wandb
                        if args.local_rank == 0:
                            # Only uncomment if you want to save model files to wandb
                            # wandb.save(best_output_model_file)
                            pass

                logger.info("Current best model is from epoch %d with R1: %.4f",
                            best_epoch, best_score)

        # Calculate and log total training time
        total_training_time = time.time() - start_time
        if args.local_rank == 0 and args.use_wandb:
            wandb.run.summary["total_training_time"] = total_training_time
            wandb.run.summary["training_time_hrs"] = total_training_time / 3600
            logger.info("Total training time: %.2f hours", total_training_time / 3600)

        ## Test on the best checkpoint
        if args.local_rank == 0:
            logger.info("Testing with the best model from epoch %d", best_epoch)
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            test_start_time = time.time()
            test_r1 = eval_epoch(args, model, test_dataloader, device, n_gpu, split="test")
            test_time = time.time() - test_start_time

            # Log final test results_batch16 and finish wandb run
            if args.use_wandb:
                wandb.run.summary["test_r1"] = test_r1
                wandb.run.summary["test_time"] = test_time
                # Create a results_batch16 table for easier viewing
                columns = ["Model", "Dataset", "Best Epoch", "Val R@1", "Test R@1", "Training Time (hrs)"]
                data = [[args.pretrained_clip_name, args.datatype, best_epoch, best_score, test_r1,
                         total_training_time / 3600]]
                results_table = wandb.Table(columns=columns, data=data)
                wandb.log({"results_batch16": results_table})

                # Finish the wandb run
                wandb.finish()

    elif args.do_eval:
        if args.local_rank == 0:
            test_start_time = time.time()
            test_r1 = eval_epoch(args, model, test_dataloader, device, n_gpu, split="test")
            test_time = time.time() - test_start_time

            logger.info("Evaluation completed with R1: %.4f in %.2f seconds",
                        test_r1, test_time)

            # Log evaluation results_batch16 and finish wandb run
            if args.use_wandb:
                wandb.run.summary["test_r1"] = test_r1
                wandb.run.summary["test_time"] = test_time
                wandb.finish()


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        # Log exception to wandb if it's initialized
        if 'args' in locals() and hasattr(args, 'use_wandb') and args.use_wandb and \
                'wandb' in globals() and wandb.run is not None:
            wandb.run.summary["error"] = str(e)
            wandb.finish(exit_code=1)
        # Re-raise the exception
        print(f"Exception occurred during execution: {str(e)}")
        raise
