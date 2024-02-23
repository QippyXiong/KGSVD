import os
import json
import pickle
from os import PathLike
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Any

import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import KGSVD, HyperParamConfig, TrainParamConfig, BayesianPersonalizedRankLoss
from metrics import compute_metrics_by_answers
from svd_torch import SVD


@torch.no_grad()
def valid_KGSVD(
    model: KGSVD,
    valid_set: np.ndarray,
    pred_answers: dict,
    K: int,
    batch_size: int,
    device: torch.device = torch.device('cuda:0'),
):
    model.to(device)
    model.eval()

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    pred_scores = []

    for batch in tqdm(valid_loader, "validating"):
        user_ids = batch[:, 0]
        item_ids = batch[:, 1]

        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        item_ids = item_ids.unsqueeze(1)

        step_pred_scores = model.forward(user_ids, item_ids, is_train=False)
        pred_scores.append(step_pred_scores.squeeze(-1).cpu().detach().numpy())
    
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_scores = np.stack([ valid_set[:, 0], valid_set[:, 1], pred_scores ], axis=1)

    return compute_metrics_by_answers(pred_scores, pred_answers, K)


def inbatch_train_single_epoch(
    model: KGSVD,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device = torch.device('cuda:0'),
) -> None:
    
    model.to(device)
    model.train()
    optimizer.zero_grad()
    loss_func = BayesianPersonalizedRankLoss()
    epoch_loss_values = []
    for user_ids, item_ids in tqdm(loader, "training"):
        if user_ids.shape[0] <= 5:
            continue

        pos_input_items = item_ids.unsqueeze(1)
        # get negative result
        neg_input_items = []
        for i in range(len(item_ids)):
            neg_sample = torch.concat((item_ids[:i], item_ids[i+1:]), dim=-1)
            neg_input_items.append(neg_sample)  # as the others are negative
        neg_input_items = torch.stack(neg_input_items)

        user_ids = user_ids.to(device)
        pos_input_items = pos_input_items.to(device)
        neg_input_items = neg_input_items.to(device)

        positive_scores = model(user_ids, pos_input_items)  # try
        negative_scores = model(user_ids, neg_input_items)
        
        loss = loss_func(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss_values.append(loss.item())
    
    return epoch_loss_values


def save_KGSVD(model, path: PathLike):
    save_path = Path(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_bin_path = save_path/"model.bin"
    torch.save(model.state_dict(), model_bin_path)
    with open(save_path/"config.json", "w") as config_fp:
        json.dump(asdict(model.config), config_fp)
    

def load_KGSVD(path: PathLike, loader) -> KGSVD:
    load_path = Path(path)
    with open(load_path/"config.json", "r") as config_fp:
        config = HyperParamConfig(**json.load(config_fp))
    model = KGSVD(config, loader)
    model.load_state_dict(torch.load(load_path/"model.bin"))
    return model


def train_KGSVD(
    model: KGSVD,
    loader: DataLoader,
    train_param: TrainParamConfig,
    test_batch_size: int,
    valid_set: np.ndarray,
    valid_answers: dict[int, list[int]],
    test_set: np.ndarray,
    test_answers: dict[int, list[int]],
    K: int,
    device: torch.device = torch.device('cuda:0'),
    check_points_dir = '.checkpoints',
    valid_span: int = 10,
    begin_epoch: int = 1,
    all_loss_values = None,
    all_epoch_metrics = None,
) -> None:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=train_param.lr)

    base_name = f'{train_param.lr}-{train_param.batch_size}-{model.config.hidden_size}' + " KGSVDw"  # w means ignore used
    base_dir = Path(check_points_dir)/base_name 
    save_model_dir = base_dir/"models"

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # save hyper param config and training config
    with open(base_dir/"hyper_param.json", "w") as config_fp:
        json.dump(asdict(model.config), config_fp)
    with open(base_dir/"train_param.json", "w") as train_param_fp:
        json.dump(asdict(train_param), train_param_fp)
    
    if all_loss_values is None:
        all_loss_values = []
    if all_epoch_metrics is None:
        all_epoch_metrics = []
    
    best_recall = 0.
    best_model_save_dir = None
    for epoch in range(begin_epoch, train_param.num_epochs+1):
        epoch_model_name = "epoch_" + str(epoch)
        epoch_model_save_dir = save_model_dir/epoch_model_name
        
        loss_values = inbatch_train_single_epoch(model, loader, optimizer, device)
        mean_loss_value = {np.mean(loss_values)}
        print(f"Epoch {epoch}, mean loss: {mean_loss_value}")

        save_KGSVD(model, epoch_model_save_dir)
        if epoch % valid_span == 0:
            # valid model for this epoch
            metrics = valid_KGSVD(model, valid_set, valid_answers, K, test_batch_size, device)
            metric_dict, mean_precison, mean_recall, mean_hit_rate = metrics
            print(f"Epoch {epoch}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")

            # compare select the best model
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_model_save_dir = epoch_model_save_dir

            all_epoch_metrics.append(metrics)

        # append the loss and metrics
        all_loss_values.append(mean_loss_value)

    # select the best valid model for test
    best_kg_svd = load_KGSVD(best_model_save_dir, model.loader)

    test_metrics = valid_KGSVD(best_kg_svd, test_set, test_answers, K, test_batch_size, device)
    metric_dict, mean_precison, mean_recall, mean_hit_rate = test_metrics
    print(f"Best model {best_model_save_dir}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")

    # save all training records
    with open(base_dir/"all_loss_values.pkl", "wb") as loss_fp:
        pickle.dump(all_loss_values, loss_fp)
    
    with open(base_dir/"all_valid_epoch_metrics.pkl", "wb") as metric_fp:
        pickle.dump(all_epoch_metrics, metric_fp)

    with open(base_dir/'test_metrics.pkl', "wb") as test_metrics_dp:
        pickle.dump(test_metrics, test_metrics_dp)
    
    with open(base_dir/'best_result', mode='w') as best_fp:
        best_fp.write(f"Best model {best_model_save_dir}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")


def save_svd(model: SVD, path: PathLike):
    save_path = Path(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_bin_path = save_path/"model.bin"
    torch.save(model.state_dict(), model_bin_path)
    with open(save_path/"config.json", "w") as config_fp:
        json.dump({ 'num_users': model.num_users, 'num_items': model.num_items, 'embed_size': model.embed_size }, config_fp)


def load_svd(path: PathLike) -> SVD:
    load_path = Path(path)
    with open(load_path/"config.json", "r") as config_fp:
        config = json.load(config_fp)
        model = SVD(config['num_users'], config['num_items'], config['embed_size'])
    model.load_state_dict(torch.load(load_path/"model.bin"))
    return model


def train_svd_single_epoch(
    model: SVD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device('cuda:0'),
) -> float:
    model.to(device)
    model.train()
    optimizer.zero_grad()
    loss_func = BayesianPersonalizedRankLoss()
    epoch_loss_values = []
    for user_ids, item_ids in tqdm(loader, "training"):
        if user_ids.shape[0] <= 5:
            continue

        pos_input_items = item_ids.unsqueeze(1)
        # get negative result
        neg_input_items = []
        for i in range(len(item_ids)):
            neg_sample = torch.concat((item_ids[:i], item_ids[i+1:]), dim=-1)
            neg_input_items.append(neg_sample)  # as the others are negative
        neg_input_items = torch.stack(neg_input_items)

        user_ids = user_ids.to(device)
        pos_input_items = pos_input_items.to(device)
        neg_input_items = neg_input_items.to(device)

        pos_ratins = model.forward(user_ids, pos_input_items)
        user_ids = user_ids.unsqueeze(-1).repeat(1, neg_input_items.shape[1])
        neg_ratings = model.forward(user_ids, neg_input_items)

        loss = loss_func(pos_ratins, neg_ratings)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss_values.append(loss.item())
    
    return np.mean(epoch_loss_values)


def train_svd(
    model: SVD,
    loader: DataLoader,
    train_param: TrainParamConfig,
    test_batch_size: int,
    valid_set: np.ndarray,
    valid_answers: dict[int, list[int]],
    test_set: np.ndarray,
    test_answers: dict[int, list[int]],
    K: int,
    device: torch.device = torch.device('cuda:0'),
    check_points_dir = '.checkpoints_svd',
    valid_span: int = 10,
    begin_epoch: int = 1
):
    optimizer = AdamW(model.parameters(), lr=train_param.lr, weight_decay=0.001)

    base_name = f'{train_param.lr}-{train_param.batch_size}-{model.embed_size}' + " SVD"
    base_dir = Path(check_points_dir)/base_name 
    save_model_dir = base_dir/"models"

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # save hyper param config and training config
    with open(base_dir/"hyper_param.json", "w") as config_fp:
        json.dump({ 'num_users': model.num_users, 'num_items': model.num_items, 'embed_size': model.embed_size }, config_fp)
    with open(base_dir/"train_param.json", "w") as train_param_fp:
        json.dump(asdict(train_param), train_param_fp)
    
    all_loss_values = []
    all_epoch_metrics = []
    
    best_recall = 0.
    best_model_save_dir = None

    model.to(device)
    for epoch in range(1, 1+train_param.num_epochs):
        epoch_model_name = "epoch_" + str(epoch)
        epoch_model_save_dir = save_model_dir/epoch_model_name

        loss_value = train_svd_single_epoch(model, loader, optimizer, device)
        all_loss_values.append(loss_value)

        print(f"epoch {epoch}, loss: { loss_value }")

        save_svd(model, epoch_model_save_dir)
        
        if epoch % valid_span == 0:
            # valid model for this epoch
            pred_scores = model.predict(valid_set[:, :2], test_batch_size)
            metrics = compute_metrics_by_answers(pred_scores, valid_answers, K)
            metric_dict, mean_precison, mean_recall, mean_hit_rate = metrics
            print(f"Epoch {epoch}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")

            # compare select the best model
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_model_save_dir = epoch_model_save_dir

            all_epoch_metrics.append(metrics)

    # select the best valid model for test
    best_svd = load_svd(best_model_save_dir)
    best_svd.to(device)
    pred_scores = best_svd.predict(test_set[:, :2], test_batch_size)
    test_metrics = compute_metrics_by_answers(pred_scores, test_answers, K)
    metric_dict, mean_precison, mean_recall, mean_hit_rate = test_metrics
    print(f"Best model {best_model_save_dir}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")

        # save all training records
    with open(base_dir/"all_loss_values.pkl", "wb") as loss_fp:
        pickle.dump(all_loss_values, loss_fp)
    
    with open(base_dir/"all_valid_epoch_metrics.pkl", "wb") as metric_fp:
        pickle.dump(all_epoch_metrics, metric_fp)

    with open(base_dir/'test_metrics.pkl', "wb") as test_metrics_dp:
        pickle.dump(test_metrics, test_metrics_dp)
    
    with open(base_dir/'best_result', mode='w') as best_fp:
        best_fp.write(f"Best model {best_model_save_dir}, recall@{K}: {mean_recall}, precision@{K}: {mean_precison}, hit_rate@{K}: {mean_hit_rate}")


def read_checkpoints_metrics(
    check_point_dir: PathLike  
) -> tuple[ list, list, Any]:
    base_dir = Path(check_point_dir)

    with open(base_dir/"all_loss_values.pkl", "rb") as loss_fp:
        all_loss_values = pickle.load(loss_fp)
    
    with open(base_dir/"all_valid_epoch_metrics.pkl", "rb") as metric_fp:
        all_epoch_metrics = pickle.load(metric_fp)

    with open(base_dir/'test_metrics.pkl', "rb") as test_metrics_dp:
        test_metrics = pickle.load(test_metrics_dp)

    return all_loss_values, all_epoch_metrics, test_metrics


def resume_train_KGSVD(
    check_point_dir: PathLike, 
    begin_epoch: int, 
    new_num_epochs: int, 
    loader, 
    test_batch_size,
    valid_span,
    K: int,
    device: torch.device = torch.device('cuda:0'),
):
    check_point_dir = Path(check_point_dir)
    model_dir = check_point_dir/'models'/f'epoch_{begin_epoch}'

    model = load_KGSVD(model_dir, loader)
    
    with open(check_point_dir/"train_param.json", "r") as train_param_fp:
        train_param = TrainParamConfig(**json.load(train_param_fp))
    
    train_param.num_epochs = new_num_epochs

    all_loss_values, all_epoch_metrics, test_metrics = read_checkpoints_metrics(check_point_dir)
    
    train_KGSVD(
        model,
        loader,
        train_param,
        test_batch_size,
        loader.get_valid_set(),
        loader.get_valid_answers(),
        loader.get_test_set(),
        loader.get_test_answers(),
        K,
        device,
        '.checkpoints',
        valid_span,
        begin_epoch,
        all_loss_values,
        all_epoch_metrics
    )
    

def resume_train_svd(
    check_point_dir: PathLike, 
    begin_epoch: int, 
    new_num_epochs: int, 
    loader, 
    test_batch_size,
    valid_span,
    K: int,
    device: torch.device = torch.device('cuda:0'),
):
    check_point_dir = Path(check_point_dir)
    model_dir = check_point_dir/'models'/f'epoch_{begin_epoch}'

    model = load_svd(model_dir)
    
    with open(check_point_dir/"train_param.json", "r") as train_param_fp:
        train_param = TrainParamConfig(**json.load(train_param_fp))
    
    all_loss_values, all_epoch_metrics, test_metrics = read_checkpoints_metrics(check_point_dir)
    
    train_svd(
        model,
        loader,
        train_param,
        test_batch_size,
        loader.valid_set,
        loader.get_valid_answers(),
        loader.test_set,
        loader.get_test_answers(),
        K,
        device,
        '.checkpoints_svd',
        valid_span,
        begin_epoch,
        new_num_epochs
    )
