#!/usr/bin/env python
# coding: utf-8

"""
Load a trained speaker and images/data to create (sample) captions for them.
The MIT License (MIT)
Originally created at 10/3/20, for Python 3.x by Panos Achlioptas 
(Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab)

Script modified to allow for modular loading
The Apache License 2.0 (Apache-2.0)
Modified at 11/05/22, for Python 3.x by Spuun
(Copyright (c) 2020 Spuun (kek@spuun.art) & Art Union)
"""


import torch
import json
import numpy as np
import pandas as pd

from artemis.in_out.arguments import parse_test_speaker_arguments
from artemis.in_out.neural_net_oriented import torch_load_model, load_saved_speaker, seed_torch_code
from artemis.neural_models.attentive_decoder import negative_log_likelihood
from artemis.captioning.sample_captions import versatile_caption_sampler, captions_as_dataframe
from artemis.in_out.datasets import sub_index_affective_dataloader
from artemis.in_out.datasets import custom_grounding_dataset_similar_to_affective_loader


def query(data):
    # Define a custom dataset to allow for loading images on-the-go
    def replacement_custom_dataset(data):
        df = df.to

    # Load pretrained speaker & its corresponding train-val-test data. If you do not provide a
    # custom set of images to annotate. Then based on the -split you designated it will annotate this data.
    speaker, epoch, data_loaders = load_saved_speaker(args.speaker_saved_args, args.speaker_checkpoint,
                                                      with_data=True, verbose=True)
    device = torch.device(
        "cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    speaker = speaker.to(device)
    eos = speaker.decoder.vocab.eos
    working_data_loader = data_loaders[args.split]

    if args.max_utterance_len is None:
        # use the maximum length in the underlying split.
        def utterance_len(tokens, eos=eos):
            # -1 to remove sos
            return np.where(np.asarray(tokens) == eos)[0][0] - 1
        args.max_utterance_len = working_data_loader.dataset.tokens.apply(
            utterance_len).max()

    use_custom_dataset = False
    if args.custom_data_csv is not None:
        use_custom_dataset = True

    if args.compute_nll and not use_custom_dataset:
        print('Computing Negative Log Likelihood of ground-truth annotations:')
        nll = negative_log_likelihood(speaker, working_data_loader, device)
        print('{} NLL: {}'.format(args.split, nll))

    if use_custom_dataset:
        annotate_loader = custom_grounding_dataset_similar_to_affective_loader(args.custom_data_csv,
                                                                               working_data_loader, args.n_workers)

    if args.subsample_data != -1:
        sids = np.random.choice(
            len(annotate_loader.dataset.image_files), args.subsample_data)
        annotate_loader = sub_index_affective_dataloader(annotate_loader, sids)

    with open(args.sampling_config_file) as fin:
        sampling_configs = json.load(fin)

    print('Loaded {} sampling configurations to try.'.format(len(sampling_configs)))
    # if you did not specify them in the sampling-config
    optional_params = ['max_utterance_len', 'drop_unk', 'drop_bigrams']
    # those from the argparse will be used
    final_results = []
    for config in sampling_configs:
        for param in optional_params:
            if param not in config:
                config[param] = args.__getattribute__(param)
        print('Sampling with configuration: ', config)

        if args.random_seed != -1:
            seed_torch_code(args.random_seed)

        captions_predicted, attn_weights = versatile_caption_sampler(
            speaker, annotate_loader, device, **config)
        df = captions_as_dataframe(
            annotate_loader.dataset, captions_predicted, wiki_art_data=not use_custom_dataset)
        final_results.append([config, df, attn_weights])
        print('Done.')

    #pickle_data(args.out_file, final_results)
    return df.to_dict()  # return a dict
