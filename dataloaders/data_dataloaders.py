import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
from dataloaders.dataloader_vatex_neg_aug_word_phrase_retrieval import VATEX_TrainDataLoader


def dataloader_vatex_train(args, tokenizer):
    vatex_dataset = VATEX_TrainDataLoader(
        subset="train",
        data_path=args.data_path,
        output_dir=args.output_dir,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        do_neg_aug=args.do_neg_aug,
        neg_aug_num_sentences=args.neg_aug_num_sentences,
        train_path_from_data_folder=args.train_path_from_data_folder,
        test_path_from_data_folder=args.test_path_from_data_folder,
        val_path_from_data_folder=args.val_path_from_data_folder,
        captions_path_from_data_folder=args.captions_path_from_data_folder
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatex_dataset), train_sampler

def dataloader_vatex_test(args, tokenizer, subset="test"):
    vatex_testset = VATEX_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        train_path_from_data_folder=args.train_path_from_data_folder,
        test_path_from_data_folder=args.test_path_from_data_folder,
        val_path_from_data_folder=args.val_path_from_data_folder,
        captions_path_from_data_folder=args.captions_path_from_data_folder
    )
    dataloader_msrvtt = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(vatex_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_train, "val":dataloader_vatex_test, "test":dataloader_vatex_test}

