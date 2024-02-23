from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
# 


class SVD(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_size: int
    ) -> None:
        super(SVD, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = embed_size

        self.user_embed = nn.Embedding(num_users, embed_size)
        self.item_embed = nn.Embedding(num_items, embed_size)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.num_users = num_users
        self.num_items = num_items

        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_embeds = self.user_embed(user_ids)
        item_embeds = self.item_embed(item_ids)

        user_bias = self.user_bias(user_ids).squeeze(-1)
        item_bias = self.item_bias(item_ids).squeeze(-1)

        scores = torch.linalg.vecdot(user_embeds, item_embeds) + user_bias + item_bias
        return scores
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        total_loss = 0.
        for user_ids, item_ids, ratings in train_loader:
            ratings = ratings.float().to(self.device)
            user_ids = user_ids.int().to(self.device)
            item_ids = item_ids.int().to(self.device)
            optimizer.zero_grad()
            predictions = self(user_ids, item_ids)
            loss = F.mse_loss(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    @torch.no_grad()
    def predict(
        self,
        user_item_ids: Union[np.ndarray, torch.Tensor],
        batch_size: int = 100 
    ) -> np.ndarray:
        self.eval()
        scores = []
        for i in range(0, len(user_item_ids), batch_size):
            input_user_ids = user_item_ids[i:i+batch_size, 0].astype(np.int32)
            input_item_ids = user_item_ids[i:i+batch_size, 1].astype(np.int32)
            input_user_ids = torch.from_numpy(input_user_ids).to(self.device)
            input_item_ids = torch.from_numpy(input_item_ids).to(self.device)
            batch_scores = self(input_user_ids, input_item_ids)
            scores.append(batch_scores)
        scores_np = torch.concat(scores, 0).cpu().numpy().astype(np.float32)
        pred_result = np.stack((user_item_ids[:, 0].astype(np.float32), user_item_ids[:, 1].astype(np.float32), scores_np), axis=-1)
        return pred_result
