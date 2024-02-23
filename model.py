from typing import Union, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np
from datetime import datetime
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

# 


class Iloader:
    r""" Interface of Model loader """
    def get_neighbors(
        self, 
        entity_ids: list[list[int]], 
        return_tensor: bool = True,
        pad_r_id: Optional[int] = None,
        pad_o_id: Optional[int] = None
    ) -> Union[
            tuple[Tensor, Tensor, Tensor], 
            tuple[list[list[list[int]]], list[list[list[int]]], list[list[list[int]]]]
        ]:
        r"""
        Args:
            entity_ids: entities needs to get the neighbours.
            input shape: (num_users, num_entities)
        Returns:
            0 neighbour_ids: shape(num_users, num_entities, num_neighbours)
            1 relation_ids: shape(num_users, num_entities, num_neighbours)
            2 neighbour_masks: True for masked, shape(num_users, num_entities, num_neighbours)
        """
        raise NotImplementedError("method get_neighbors must be overrided")
    
    def read_embeddings(self, num_entities: int) -> Tensor:
        r"""
        Args:
            num_entities: number of entity embeddings to read.
        
        Return:
            entity_embeddings: embeddings for each entity, shape(num_entities, embed_size)
        """
        raise NotImplementedError("method read_embeddings must be overrided")
    
    def save_embeddings(self, embeddings: Tensor) -> None:
        r"""
        Args:
            embeddings: embeddings to save, shape(num_entities, embed_size)
        """
        raise NotImplementedError("method save_embeddings must be overrided")
    
    def read_interacted_items(
        self, 
        user_ids: list[int],
        ignore_item_ids: Optional[list[list[int]]] = None,
        pad_item_id: Optional[int] = None,
        return_tensor: bool = True
    ) -> tuple[Tensor, Tensor] | tuple[list[list[int]], list[list[int]]]:
        r"""
        read user iteracted items, and ignore some items.
        
        Args:
            user_ids: array of id of user for get iteracted items, shape(num_users)
            ignore_item_ids: each item is the ignore item id for each user, shape(num_users, num_ignores)
            pad_item_id: the pad item id for padding

        Returns:
            0 interacted_item_ids: shape(num_users, num_interacted_items)
            1 interacted_item_masks: True for masked, shape(num_users, num_interacted_items)
        """
        raise NotImplementedError("method read_interacted_items must be overrided")
    
    def global_collecting(self, entity_ids: list[int], num_paths: int, num_steps: int, p: float, num_collect: Union[int, list[int]]) -> Tensor:
        r"""
        using random walk to collect global context, implement rely on self.get_neighbors
        default num_collect is set to the size of neighbors set

        Returns:
            0 paths: shape(num_entities, num_paths, num_steps)
            1 end_node_ids: list[list[int]] shape(num_entities, num_collect)
        """
        if isinstance(num_collect, int):
            num_collect = [num_collect] * len(entity_ids)
        paths: list[list[tuple[int, int]]] = [ [] ] * len(entity_ids)
        r"""
        collect one path logic:
        cur_entity_id := $input_entity_id
        path = []
        1. get all neighbors of each entity
        2. while len(path) < self.num_steps:
            1) neighbours = self.get_neighbors(cur_entity_id)
            2) 
        """
        for _ in range(num_paths):
            for i, entity_id in enumerate(entity_ids):
                path: list[tuple[int, int]] = []
                cur_id = entity_id
                while len(path) < num_steps: # collect path
                    nei_ids, rel_ids, _ = self.get_neighbors([cur_id], False)
                    for nei_id, rel_id in zip(nei_ids[0], rel_ids[0]):
                        if nei_id == entity_id:  # should not collect itself
                            continue
                        prob = p if (nei_id, rel_id) in path else 1 - p
                        if np.random.rand() < prob:
                            path.append((nei_id, rel_id))
                            cur_id = nei_id
                            break
                paths[i].append(path)
        
        entity_with_freq = []
        for entity_paths in paths:
            path = []
            for t_arr in entity_paths:
                path.extend(t_arr)
            entity_freq = [
                (e_id, len([ 1 for e, _ in path if e == e_id ]))
                for e_id, _ in path
            ]
            entity_freq.sort(key=lambda x: x[1], reverse=True)
            entity_with_freq.append(entity_freq)
        min_entity_nums = min([ len(arr) for arr in entity_with_freq ])
        top_k = min(min_entity_nums, num_collect)
        top_k_freq_entities = [ e_id for e_id, _ in entity_freq[:top_k] ]
        return paths, top_k_freq_entities
    
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("method __getitem__ must be overrided")


@dataclass
class HyperParamConfig:
    # embedding nums
    num_users: int
    num_items: int
    num_entities: int
    num_relations: int

    # vector sizes
    entity_embed_size: int
    relation_embed_size: int
    user_embed_size: int
    hidden_size: int

    # random collect params
    # gamma: float     # random walk start probability
    # num_paths: int   # random walk paths
    # num_steps: int # one path collect nums

    dropout: float


@dataclass
class TrainParamConfig:
    lr: float
    num_epochs: int
    # dropout: float
    weight_decay: float
    batch_size: int = 24



class LocalAttentionLayer(nn.Module):
    r"""
    Attention layer for Item local context pooling
    """

    def __init__(self, dropout: float, mask_value: float = -1e7) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mask_value = mask_value

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Args:
            queries: input queries for attention pooling
            keys: input keys for attention pooling
            values: input values for attention pooling
            attention_mask: False means mask this key/value

        Shapes:
            queries: (L, E)
            keys: ([L,] M, S, E)
            values: ([L,] M, S, Ev)
            attention_mask: (L, M, S)
        Returns:
            0: attention_pooling_result: (L, M, Ev)
            1: attention_weights: (L, M, S)
        """

        r"""
        compute logic:
        1. weights shape should be like (L, M, S), so weights computed by
            1) repeat queries to (L, M, S, E)
            2) repeat keys to (L, M, S, E)
            3) compute attention weights softmax( queries \dot keys )
            4) repeat values to (L, M, S, Ev)
            5) repeat weights to (L, M, 1, S)
            6) compute attention pooling result = (weights @ values)
        """
        size_L = queries.shape[0]
        size_M = keys.shape[-3]
        size_S = keys.shape[-2]

        input_queries = torch.repeat_interleave(
            queries.unsqueeze(-2), 
            size_M, 
            dim=-2, 
            output_size=size_M
        )  # (L, M, E)

        input_queries = torch.repeat_interleave(
            input_queries.unsqueeze(-2), 
            size_S, 
            dim=-2, 
            output_size=size_S
        )  # (L, M, S, E)

        if keys.dim() == 3:
            input_keys = torch.repeat_interleave(
                keys.unsqueeze(0), 
                size_L, 
                dim=0, 
                output_size=size_L
            )  # (L, M, S, E)
        else:
            input_keys = keys

        attn_scores = torch.linalg.vecdot(input_queries, input_keys)  # (L, M, S)
        attn_scores = attn_scores + (~attention_mask).float() * self.mask_value
        attn_weights = F.softmax(attn_scores, dim=-1)  # (L, M, S)
        
        attn_weights= attn_weights.unsqueeze(-2) # (L, M, 1, S)

        if values.dim() == 3:
            input_values = torch.repeat_interleave(
                values.unsqueeze(0), 
                size_L, 
                dim=0, 
                output_size=size_L
            )
        else:
            input_values = values

        attention_pooling_result = attn_weights @ input_values  # (L, M, 1, Ev)
        attention_pooling_result = attention_pooling_result.squeeze(-2)  # (L, M, Ev)
        attention_pooling_result = self.dropout(attention_pooling_result) # dropout
        return attention_pooling_result, attn_weights.squeeze(-2)


class LocalAttentionPoolingModel(nn.Module):
    r"""
    Item Local KG Context Pooling Model
    """

    def __init__(self, config: HyperParamConfig) -> None:
        super().__init__()
        self.config = config

        self.user_mapper = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.user_embed_size, config.hidden_size),
            nn.ReLU(),
        )

        self.neighbour_mapper = nn.Linear(config.relation_embed_size + config.entity_embed_size, config.hidden_size, False)
        self.key_mapper = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.relation_embed_size + config.entity_embed_size, config.hidden_size),
            nn.Tanh(),
        )

        self.attn = LocalAttentionLayer(self.config.dropout)

        self.mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size + config.entity_embed_size, config.hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        user_embeds: Tensor,
        target_item_embeds: Tensor,
        relation_embeds: Tensor,
        neighbor_embeds: Tensor,
        attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        @TODO: support item_embeds for shape (num_users, num_items, entity_embed_size)

        Shapes:
            user_embeds: (num_users, user_embed_size), num_users is like num_queries
            target_item_embeds: (num_users, num_items, entity_embed_size), 
            relation_embeds: (num_users, num_items, num_neighbours, relation_embed_size) 
            neighbor_embeds: (num_users, num_items, num_neighbours, entity_embed_size)
            attention_mask: (num_users, num_items, num_neigbours)
        
        Returns:
            0 local_context: (num_users, num_items, hidden_size)
            1 attn_weights: (num_users, num_items, num_neigbours)
            2 attn_keys: (num_users, num_items, num_neigbours, hidden_size)
        """
        num_neighbours = neighbor_embeds.shape[-2]
        input_user_embeds = self.user_mapper(user_embeds)  # (num_users, hidden_size)

        input_target_item_embeds = target_item_embeds.unsqueeze(-2).repeat(1, 1, num_neighbours, 1)
        input_neighbour_embeds = torch.concat((relation_embeds, input_target_item_embeds), dim=-1)  

        input_local_embeds = self.neighbour_mapper(input_neighbour_embeds)
        # (num_items, num_neigbours, hidden_size)

        input_target_item_local_embeds = torch.concat((input_local_embeds, input_target_item_embeds), dim=-1)
        input_local_keys = self.key_mapper(input_target_item_local_embeds)
        # (num_items, num_neigbours, hidden_size)

        # repeat mask to (num_users, num_items, num_neigbours)
        # attention_mask = attention_mask.unsqueeze(0).repeat(input_user_embeds.shape[0], 1, 1)

        pooling_output, attn_weights = self.attn.forward(
            input_user_embeds,
            input_local_keys,
            neighbor_embeds,
            attention_mask,
        )  # (num_users, num_items, hidden_size), (num_users, num_items, num_neigbours)

        input_mlp = torch.concat((pooling_output, target_item_embeds), dim=-1)  
        # (num_users, num_items, hidden_size + entity_embed_size)
        local_context = self.mlp(input_mlp)
        return local_context, attn_weights, input_local_keys  # (num_users, num_items, hidden_size)


class ItemContextPooler(nn.Module):
    r"""
    Utilizing local pooler and global pooler for item context pooling
    """

    def __init__(self, config: HyperParamConfig, use_global: bool = False) -> None:
        super().__init__()
        
        self.config = config
        self.use_global = use_global
        
        self.local_pooler = LocalAttentionPoolingModel(config)
        if self.use_global:
            self.global_pooler = nn.GRU(config.entity_embed_size * 2, config.hidden_size, 1, batch_first=True, dropout=config.dropout)
        self.gate_weight = nn.Parameter(torch.randn(config.hidden_size), requires_grad=True)

    def forward(
        self,
        user_embeds: Tensor,
        item_embeds: Tensor,
        neighbour_embeds: Tensor,
        relation_embeds: Tensor,
        neighbour_masks: Tensor,
        global_embeds: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        r"""
        Context pooling for each user and for each item in list of items
        each user for per batch of items
        @TODO: support item_embeds for shape (num_users, num_items, entity_embed_size)

        Shapes:
            user_embeds: (num_users, user_embed_size)
            item_embeds: ([num_users ,] num_items, entity_embed_size)
            neighbour_embeds: ([num_users ,] num_items, num_neighbours, entity_embed_size)
            relation_embeds: ([num_users ,] num_items, num_neighbours, relation_embed_size)
            neighbour_masks: ([num_users ,] num_items, num_neighbours)
            global_embeds: ([num_users ,] num_items, num_collect, entity_embed_size)

        Return:
            0 item_repr: (num_users, num_items, entity_embed_size)
            1 item_local_attn_weights: (num_users, num_items, num_neighbours)
            2 item_er_presents: (num_items, entity_embed_size)
        """
        num_users = user_embeds.shape[0]

        local_context, local_attention_weights, er_presents = self.local_pooler.forward(
            user_embeds,
            item_embeds,
            relation_embeds,
            neighbour_embeds,
            neighbour_masks
        )

        if self.use_global:
            if input_global_embeds is None:
                raise ValueError("global_embeds should not be None when use_global is True")
            
            num_collect = global_embeds.shape[-2]

            input_item_embeds = item_embeds.unsqueeze(-2).repeat(1, num_collect, 1)
            # shape (num_items, num_collect, entity_embed_size)

            input_global_embeds = torch.concat((input_item_embeds, global_embeds), dim=-1)
            # shape (num_items, num_collect, entity_embed_size * 2)

            global_context, _ = self.global_pooler(input_global_embeds)
            # shape (num_items, hidden_size)

            # use gate merge two context
            gate_weights = F.sigmoid(self.gate_weight)
            context = gate_weights * local_context + (1 - gate_weights) * global_context
        else:
            context = local_context
        
        item_repr = torch.concat((item_embeds, context), dim=-1)
        return item_repr, local_attention_weights, er_presents

    def init_embedding_weights_from_loader(self):
        raise NotImplementedError("method init_embedding_weights_from_loader must be overrided")

    def save_embedding_weights_to_loader(self):
        raise NotImplementedError("method save_embedding_weights must be overrided")


class UserContextAttentionPooler(nn.Module):
    r"""
    Pooling user context according to user interacted items
    """

    def __init__(self, context_size: int, user_embed_size: int, dropout: float, mask_value: float = -1e7) -> None:
        super().__init__()
        self.context_size = context_size
        self.user_embed_size = user_embed_size
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_size*2, 1),
            nn.Tanh()
        )  # for addictive attention
        self.mask_value = mask_value
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(user_embed_size + context_size, context_size),
            nn.ReLU(),
        )

    def forward(
        self,
        target_items_context: Tensor,
        interacted_items_context: Tensor,
        user_embeds: Tensor,
        attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Args:
            target_items_context: (num_users, num_items, context_size)
            interacted_items_context: (num_users, num_interacted_items, context_size)
            user_embeds: (num_users, user_embed_size)
            attention_mask: False means mask this key/value

        Shapes:
            target_items_context: (num_users, num_items, context_size)
            interacted_items_context: (num_users, num_interacted_items, context_size)
            attention_mask: (num_users, num_interacted_items)
        
        Returns:
            0: context: (num_users, num_items, context_size)
            1: attention_weights: (num_users, num_items, num_interacted_items)
        """
        num_items = target_items_context.shape[1]
        num_users = interacted_items_context.shape[0]
        num_interacted_items = interacted_items_context.shape[-2]

        input_target_item_context = target_items_context.unsqueeze(-2).repeat(1, 1, num_interacted_items, 1)
        # (num_users, num_items, num_interacted_items, context_size)

        input_interacted_items_context = interacted_items_context.unsqueeze(1).repeat(1, num_items, 1, 1)
        # (num_users, num_items, num_interacted_items, context_size)

        input_keys = torch.concat((input_target_item_context, input_interacted_items_context), dim=-1)

        attention_scores = self.dense(input_keys).squeeze(-1)

        input_attn_mask = attention_mask.unsqueeze(1).repeat(1, num_items, 1)
        attention_scores = attention_scores + (~input_attn_mask).float() * self.mask_value
        attention_weights = F.softmax(attention_scores, dim=-1)
        # (num_users, num_items, num_interacted_items)

        attention_weights = attention_weights.unsqueeze(-2)
        # (num_users, num_items, 1, num_interacted_items)

        interacted_context = attention_weights @ input_interacted_items_context
        interacted_context = interacted_context.squeeze(-2)
        # (num_users, num_items, context_size)

        input_user_embeds = user_embeds.unsqueeze(1).repeat(1, num_items, 1)
        # (num_users, num_items, user_embed_size)

        user_context = self.mlp(
            torch.concat((input_user_embeds, interacted_context), dim=-1)
        )

        return user_context, attention_weights.squeeze(-2)


class KGSVD(nn.Module):
    r"""
    !NOTICE: There is no item_ids, only entity ids, which means should convert interacted item id to entity id first
    """

    def __init__(self, config: HyperParamConfig, loader: Iloader) -> None:
        super().__init__()
        self.config = config
        self.loader = loader
        self.item_pooler = ItemContextPooler(config)
        self.user_pooler = UserContextAttentionPooler(config.hidden_size*2, config.user_embed_size, config.dropout)

        # initial embeddings
        self.entity_embedding = nn.Embedding(config.num_entities, config.entity_embed_size)
        self.user_embedding = nn.Embedding(config.num_users, config.user_embed_size)
        self.relation_embedding = nn.Embedding(config.num_relations, config.relation_embed_size)

        self.user_context_embedding = nn.Embedding(config.num_users, config.hidden_size)
        # dummy param to get device
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def device(self):
        return self.dummy_param.device

    def forward(
        self,
        user_ids: Tensor,
        target_item_ids: Tensor,
        is_train: bool = True
    ) -> Tensor:
        r"""
        So input user_ids and a batch of target item_ids for each user and return the target items_scores.
        Actually input target_item_ids is the entity_id of the target item, not the idx of item.

        Shapes:
            user_ids: (num_users)
            target_item_ids: (num_users, num_items)
        
        Args:
            user_ids: 
            target_item_ids: each line for target scoring items for each user

        Returns:
            0: pred user scores for each item, shape(num_users, num_items)
        """
        neighbour_ids, neighbour_relations, neighbour_masks = self.loader.get_neighbors(target_item_ids)
        
        neighbour_ids = neighbour_ids.to(self.device)
        neighbour_relations = neighbour_relations.to(self.device)
        neighbour_masks = neighbour_masks.to(self.device)
        
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.entity_embedding(target_item_ids)  # (num_users, num_items, d)
        neighbour_embeds = self.entity_embedding(neighbour_ids) # (num_users, num_items, num_neighbours, d)
        relations_embeds = self.relation_embedding(neighbour_relations) # (num_users, num_items, num_neighbours, d)

        target_items_context, attn_weights, attn_keys = self.item_pooler.forward(
            user_embeds,
            item_embeds,
            neighbour_embeds,
            relations_embeds,
            neighbour_masks,
            None
        )  # (num_items, num_collect, entity_embed_size)

        users_kg_context = self.user_context_embedding(user_ids)
        users_context = torch.concat((user_embeds, users_kg_context), dim=-1)
        users_context = users_context.unsqueeze(1).repeat(1, target_items_context.shape[1], 1)

        # user_interacted_item_ids, users_mask = self.loader.read_interacted_items(user_ids.tolist(), target_item_ids.tolist() if is_train else [])

        # user_interacted_item_ids = user_interacted_item_ids.to(self.device)
        # users_mask = users_mask.to(self.device)


        # user_interacted_item_neighbours, user_interacted_item_rels, user_item_neighbour_mask = self.loader.get_neighbors(user_interacted_item_ids, return_tensor=True)

        # user_interacted_item_neighbours = user_interacted_item_neighbours.to(self.device)
        # user_interacted_item_rels = user_interacted_item_rels.to(self.device)
        # user_item_neighbour_mask = user_item_neighbour_mask.to(self.device)

        # user_interacted_item_embeds = self.entity_embedding(user_interacted_item_ids)
        # user_interacted_item_neighbours_embeds = self.entity_embedding(user_interacted_item_neighbours)
        # user_interacted_item_rels_embeds = self.relation_embedding(user_interacted_item_rels)
        
        # interacted_items_context, _, interacted_item_attn_keys = self.item_pooler.forward(
        #     user_embeds=user_embeds,
        #     item_embeds=user_interacted_item_embeds,
        #     neighbour_embeds=user_interacted_item_neighbours_embeds,
        #     relation_embeds=user_interacted_item_rels_embeds,
        #     neighbour_masks=user_item_neighbour_mask,
        #     global_embeds=None
        # )  # shape(num_users, num_interacted_items, entity_embed_size)

        # users_context, user_attn_weights = self.user_pooler.forward(
        #     target_items_context,
        #     interacted_items_context,
        #     user_embeds,
        #     users_mask
        # )  # (num_users, num_items, context_size)

        scores = torch.linalg.vecdot(users_context, target_items_context)
        # (num_users, num_items)

        return scores
    

class BayesianPersonalizedRankLoss(nn.Module):
    r"""
    Bayesian Personalized Ranking loss function

    
    """

    def __init__(self) -> None:
        super().__init__()
        ...
    
    def forward(self, positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
        r"""
        Shapes:
            positive_scores: (num_users, num_items)
            negative_scores: (num_users, num_items, num_negative_samples)
        """
        loss = - F.logsigmoid((positive_scores - negative_scores.mean(-1)).mean())
        return loss


class KGRegularLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        ...
    
    def forward(self, entity_embedings: Tensor, entity_neighbour_embeddings: Tensor) -> Tensor:
        r"""
        Shapes:
            entity_embedings: (num_entities, d)
            entity_neighbour_embeddings: (num_entities, num_neighbours, d)
        """
        loss = F.mse_loss(entity_embedings.unsqueeze(1), entity_neighbour_embeddings, reduction='mean')
        return loss


class PersonalPageRankLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        ...

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        r"""
        Shapes:
            scores: (num_users, num_items)
            labels: (num_users, num_items) dtype=bool

        """
        input_labels = (labels.float() - 0.5) * 2  # 1 for positive, -1 for negative
        loss = (-scores * input_labels).sum()
        return loss
