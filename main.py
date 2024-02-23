from datetime import datetime
from argparse import ArgumentParser

import torch
import numpy as np
from torch.utils.data import DataLoader

from loader import load_artist_kg, UserItemInteractionLoader, KGLoader, load_artists_interactions, ComposedLoader
from model import KGSVD, HyperParamConfig, BayesianPersonalizedRankLoss, TrainParamConfig
from svd_torch import SVD
from metrics import compute_metrics_by_answers
from train import train_KGSVD, valid_KGSVD, save_KGSVD, load_KGSVD, train_svd, resume_train_KGSVD, resume_train_svd

def tarin_kgsvd_main(args):
    dim = args.dim
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr

    valid_span = args.valid_span
    weight_decay = 1e-3
    dropout = 0.

    item_id2entity_id, num_entities, triplets, relatioins = load_artist_kg()
    include_items = set(item_id2entity_id.keys())
    users, items, interactions = load_artists_interactions(include_items=include_items)
    loader = ComposedLoader(
        item_id2entity_id, num_entities, triplets, relatioins, True, users, items, interactions
    )
    config = HyperParamConfig(
        num_users=loader.num_users, 
        num_items=loader.num_items,
        num_entities=loader.num_entities,
        num_relations=loader.num_relations, 
        entity_embed_size= dim,
        relation_embed_size = dim,
        user_embed_size = dim,
        hidden_size = dim,
        dropout=dropout
    )
    train_args = TrainParamConfig(
        lr=lr,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        batch_size=batch_size
    )

    if args.checkpoint_dir:
        model = load_KGSVD(args.checkpoint_dir + f'/models/epoch_{args.resume_epoch}', loader)
    else:
        model = KGSVD(config, loader)
    data_loader = DataLoader(loader, batch_size, shuffle=True, num_workers=3)
    
    begin_epoch = 1 if not args.checkpoint_dir else args.resume_epoch + 1
    
    train_KGSVD(
        model, 
        data_loader, 
        train_args,
        18,
        loader.get_valid_set(),
        loader.get_valid_answers(),
        loader.get_test_set(),
        loader.get_test_answers(),
        K=10,
        device=torch.device('cuda:1'),
        valid_span=valid_span,
        begin_epoch=begin_epoch,
    )


def train_svd_main(args):
    train_parms = TrainParamConfig(
        lr=args.lr,
        num_epochs=args.num_epochs,
        weight_decay=1e-3,
        batch_size=args.batch_size
    )
    test_batch_szie = 100

    item_id2entity_id, num_entities, triplets, relatioins = load_artist_kg()
    include_items = set(item_id2entity_id.keys())
    users, items, interactions = load_artists_interactions(include_items=include_items)
    loader = UserItemInteractionLoader(users, items, interactions)

    svd = SVD(loader.num_users, loader.num_items, args.dim)
    train_loader = torch.utils.data.DataLoader(loader.pos_inters[:,:2].tolist(), batch_size=train_parms.batch_size, shuffle=True)

    train_svd(
        svd, 
        train_loader, 
        train_parms, 
        test_batch_szie, 
        loader.valid, 
        loader.valid_answers,
        loader.test,
        loader.test_answers,
        10,
        torch.device('cuda:1'),
        '.checkpoints_svd',
        valid_span=5,
    )


def kgsvd_resume_train_main(args):

    valid_span = args.valid_span

    item_id2entity_id, num_entities, triplets, relatioins = load_artist_kg()
    include_items = set(item_id2entity_id.keys())
    users, items, interactions = load_artists_interactions(include_items=include_items)
    loader = ComposedLoader(
        item_id2entity_id, num_entities, triplets, relatioins, True, users, items, interactions
    )

    resume_train_KGSVD(
        args.checkpoint_dir, 
        args.resume_epoch, 
        args.new_num_epochs, 
        loader, 
        18, 
        valid_span, 
        18
    )


def svd_resume_train_main(args):
    item_id2entity_id, num_entities, triplets, relatioins = load_artist_kg()
    include_items = set(item_id2entity_id.keys())
    users, items, interactions = load_artists_interactions(include_items=include_items)
    loader = UserItemInteractionLoader(users, items, interactions)

    resume_train_svd(
        args.checkpoint_dir,
        args.resume_epoch,
        args.new_num_epochs,
        loader,
        100,
        args.valid_span,
        torch.device('cuda:1')
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='kgsvd', type=str)
    parser.add_argument('--dim', default=8, type=int)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--valid_span', default=10, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint_dir', default='', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--new_num_epochs', default=0, type=int)
    
    args = parser.parse_args()
    if args.model == 'kgsvd':
        # if args.resume:
        #     kgsvd_resume_train_main(args)
        # else:
        tarin_kgsvd_main(args)
    elif args.model == 'svd':
        if args.resume:
            svd_resume_train_main(args)
        else:
            train_svd_main(args)
    else:
        raise ValueError(f"model {args.model} not supported")
