from os import PathLike
from pathlib import Path
from collections import defaultdict
from threading import Thread
from typing import Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
from torch import Tensor

from model import Iloader

def load_movie_kg(
    floader_path: PathLike = '/home/listalina/data/projects/db_netd/data/movie'
):
    # remove all nodes

    floader = Path(floader_path)

    kg_file             = floader/'kg.txt'
    item_id2entity_id   = floader/'item_index2entity_id.txt'
    file_id2name_file   = floader/'ml-20m'/'movies.csv'

    movie_id2name       = pd.read_csv(file_id2name_file, index_col=0)

    # parse kg
    movies = list[dict]
    movie_item_ids = []
    movie_entity_ids = []
    with open(item_id2entity_id, 'r') as f:
        for line in f:
            item_id, entity_id = line.strip().split('\t')
            item_id, entity_id = int(item_id), int(entity_id)
            movie_item_ids.append(item_id)
            movie_entity_ids.append(entity_id)
    
    movie_names = movie_id2name.loc[movie_item_ids]['title'].values
    movie_tags = movie_id2name.loc[movie_item_ids]['genres'].values
    movies = [
        # must match the property names in Movie
        { 'name': name, 'item_id': item_id, 'entity_id': entity_id, 'tags': tags }
        for name, item_id, entity_id, tags \
        in zip(movie_names, movie_item_ids, movie_entity_ids, movie_tags)
    ]
    # actually the entities_id is equal to range(0, len(movies))
    
    # read kg as triplets
    # as (s, r, o)'
    relations = []
    triplets = [] 
    with open(kg_file, 'r') as f:

        for line in f:
            s, r, o = line.strip().split('\t')
            if r not in relations:
                relations.append(r)
            r_id = relations.index(r)
            sub_idx = int(s)
            obj_idx = int(o)
            triplets.append((sub_idx, r_id, obj_idx))
    
    return movies, triplets, relations


def load_artist_kg(
    floader_path: PathLike = '/home/listalina/data/projects/db_netd/data/music'
):
    floader = Path(floader_path)

    kg_file             = floader/'kg.txt'
    item_id2entity_id   = floader/'item_index2entity_id.txt'

    artists = dict()
    with open(item_id2entity_id, 'r') as f:
        for line in f:
            item_id, entity_id = line.strip().split('\t')
            item_id, entity_id = int(item_id), int(entity_id)
            artists[item_id] = entity_id
    
    triplets = []
    relations = []
    entities = set()

    with open(kg_file, 'r') as f:
        for line in f:
            s, r, o = line.strip().split('\t')
            if r not in relations:
                relations.append(r)
            r_id = relations.index(r)
            sub_idx, obj_idx = int(s), int(o)
            entities.add(sub_idx)
            entities.add(obj_idx)
            triplets.append((sub_idx, r_id, obj_idx))

    # user_interaction = pd.read_csv(interaction_file, sep='\t')

    return artists, len(entities), triplets, relations


def load_artists_interactions(
    file_path: PathLike = '/home/listalina/data/projects/db_netd/data/music/user_artists.dat',
    include_items: Optional[set[int]] = None
):
    r"""
    Returns:
        0: users
        1: items
        2: interactions
    """
    if include_items is not None:
        if type(include_items) is not set:
            include_items = set(include_items)

    user_interaction = pd.read_csv(file_path, sep='\t')

    # build user-item matrix
    users = list(set(user_interaction['userID'].values))
    all_items = set(user_interaction['artistID'].values)
    if include_items is not None:
        accept_items = all_items & include_items
        residue = include_items - all_items
        if residue:
            print(f"Warning: {len(residue)} items not in the interaction file")
        items = list(accept_items)
    else:
        items = list(all_items)
    print(len(items))

    user_id2idx = { v: i  for i, v in enumerate(users) }
    item_id2idx = { v: i  for i, v in enumerate(items) }

    num_users = len(users)
    num_items = len(items)

    user_ids = np.arange(num_users)
    item_ids = np.arange(num_items)

    inter_matrix = np.zeros((num_users, num_items), dtype=np.int32)
    for _, row in user_interaction.iterrows():
        if include_items is not None and row['artistID'] not in include_items:
            continue
        user_idx = user_id2idx[row['userID']]
        item_idx = item_id2idx[row['artistID']]
        inter_matrix[user_idx, item_idx] = 1 if row['weight'] > 0 else 0
    
    weights = inter_matrix.ravel()
    # avg, std = np.mean(weights), np.std(weights)
    # weights = (weights - avg) / std

    user_ids = np.repeat(user_ids.reshape(num_users, 1), num_items, -1).ravel()
    item_ids = np.repeat(item_ids.reshape(1, num_items), num_users, 0).ravel()

    interactions = np.stack([user_ids, item_ids, weights], -1)

    return users, items, interactions


class KGLoader(Iloader):
    r"""
    loader of KG, implement get_neighbours
    """
    def __init__(
            self, 
            item_to_entity_id: dict[int, int],
            num_entities: int,
            triplets: list[tuple[int, int, int]], 
            relations: list[str],
            add_inv: bool = False
        ) -> None:
        self.item_id2entity_id = item_to_entity_id
        self.num_entities = num_entities
        self.triplets = np.array(triplets)
        self.relations = relations
        self.entity_id2item_id = { v: k for k, v in item_to_entity_id.items() }

        # repeat inv triplets
        if add_inv:
            num_r = len(self.relations)
            inv_relations = [ f'inv_{rel}' for rel in self.relations ]
            self.relations.extend(inv_relations)
                
            inv_triplets = np.array([(o, r + num_r, s) for s, r, o in triplets])
            self.triplets = np.concatenate([self.triplets, inv_triplets], 0)

        # add pad NULL relation and NULL entity
        self.relations.append('NULL')
        self.num_entities += 1
        
        self.neighbour_M = np.full((self.num_entities, len(self.triplets)), False, dtype=bool)
        
        # so in this way, we only regnize entity_idx
        subject_indices = self.triplets[:, 0]
        triplets_indices = np.arange(len(self.triplets))
        self.neighbour_M[subject_indices, triplets_indices] = True

        print(f"[KGLoader] num_entities: {self.num_entities}, num_relations: {len(self.relations)}, num_triplets: {len(self.triplets)}")

    def convert_item_ids_to_entity_ids(
        self, 
        item_ids: list[int], 
        pad_unk_item: bool = True, 
        pad_ent_id: Optional[int] = None
    ) -> list[int]:
        if pad_ent_id is None:
            pad_ent_id = self.pad_ent_id
        if pad_unk_item is not None:
            return [ self.item_id2entity_id.get(item_id, self.pad_ent_id) for item_id in item_ids ]
        return [ self.item_id2entity_id[item_id] for item_id in item_ids ]

    def conver_entity_ids_to_item_ids(
        self, 
        entity_ids: list[int],
    ) -> list[int]:
        return [ self.entity_id2item_id.get(entity_id, -1) for entity_id in entity_ids ]

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    @property
    def pad_rel_id(self) -> int:
        return self.num_relations - 1

    @property
    def pad_ent_id(self) -> int:
        return self.num_entities - 1

    def get_neighbors(
        self, 
        entity_ids: list[list[int]], 
        return_tensor: bool = True,
        pad_r_id: Optional[int] = None,
        pad_o_id: Optional[int] = None
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[list[list[list[int]]], list[list[list[int]]], list[list[list[int]]]]:
        r"""
        NOTICE: padding process will pad r for pad_r_id, pad o for pad_o_id
        
        Args:
            entity_ids: entities needs to get the neighbours with input shape: (num_users, num_entities)
            return_tensor: if True, return pytorch tensor, else return int lists
        Returns:
            0 neighbour_ids: shape(num_users, num_entities, num_neighbours)
            1 relation_ids: shape(num_users, num_entities, num_neighbours)
            2 neighbour_masks: True for masked, shape(num_users, num_entities, num_neighbours)
        """
        if pad_r_id is None:
            pad_r_id = self.pad_rel_id
        if pad_o_id is None:
            pad_o_id = self.pad_ent_id
        
        rt_neighbours: list[list[list[int]]] = []
        rt_relations: list[list[list[int]]] = []
        max_num = 0

        for each_user_entity_ids in entity_ids:
            single_user_rt_neighbours = []
            single_user_rt_relations = []

            for entity_id in each_user_entity_ids:
                tmp_triplets_indices = np.nonzero(self.neighbour_M[entity_id])
                rels = self.triplets[tmp_triplets_indices][:, 1]
                objs = self.triplets[tmp_triplets_indices][:, 2]

                single_user_rt_neighbours.append(objs.tolist())
                single_user_rt_relations.append(rels.tolist())
                if len(objs) > max_num:
                    max_num = len(objs)
            
            rt_neighbours.append(single_user_rt_neighbours)
            rt_relations.append(single_user_rt_relations)
        
        # padding process
        neighbour_masks = []
        for single_user_neighbours, single_user_rels in zip(rt_neighbours, rt_relations):
            single_user_nei_mask = []
            for neighbours, rels in zip(single_user_neighbours, single_user_rels):
                pad_len = max_num - len(neighbours)
                single_user_nei_mask.append([True] * len(neighbours) + [False] * pad_len)
                neighbours.extend([pad_o_id] * pad_len)
                rels.extend([pad_r_id] * pad_len)
            neighbour_masks.append(single_user_nei_mask)

        if return_tensor:
            return (
                Tensor(rt_neighbours).long(), 
                Tensor(rt_relations).long(), 
                Tensor(neighbour_masks).bool()
            )
        
        return rt_neighbours, rt_relations, neighbour_masks


class UserItemInteractionLoader(Iloader):
    r"""
    __next__: return shape (batch_size, 3) - (user_idx, item_idx, weight)
    """

    def __init__(
        self,
        users,
        items,
        interactions: np.ndarray,
        portions: tuple[float, float, float] = (0.6, 0.2,)
    ) -> tuple[np.array, dict, dict]:
        r"""
        Args:
            portions: portion of train and valid while test is 1 - train - valid
        
        """
        self.users = users
        self.items = items
        
        # split train,valid, test - 3:1:1
        num_interactions = len(interactions)
        train_portion, valid_portion = portions

        # time1 = datetime.now()
        # np.random.shuffle(interactions)  # takes a very long time
        interactions = np.random.permutation(interactions)
        # time2 = datetime.now()
        # print("END SHUFFER FOR ", time2 - time1)
        train_size = int(num_interactions * train_portion)
        valid_size = int(num_interactions * valid_portion)
        self.train = interactions[:train_size]
        self.valid = interactions[train_size:train_size+valid_size]
        self.test = interactions[train_size+valid_size:]

        print("END SPLIT")

        self.valid_answers = self.build_answer_for_user(self.valid)
        self.test_answers = self.build_answer_for_user(self.test)
        self.train_answers = self.build_answer_for_user(self.train)

        print("END BUILD ANSWER")

        self.pos_inters = self.train[self.train[:,2] > 0]
        
        print(f"[interactions] train: {len(self.pos_inters)}, valid: {len(self.valid)}, test: {len(self.test)}")

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_items(self) -> int:
        return len(self.items)
    
    @property
    def pad_item_id(self) -> int:
        return self.num_items - 1

    def __getitem__(self, index):
        return self.train[index][[0,1]].tolist()
    
    def __len__(self):
        return len(self.train)
    
    # def __next__(self) -> np.ndarray:
    #     if self.iter_idx >= len(self.train):
    #         np.random.shuffle(self.train)
    #         self.iter_idx = 0

    #     return_size = min(self.batch_size, len(self.train) - self.iter_idx)    
    #     self.iter_idx += return_size

    #     if self.batch_size == 1:
    #         return self.train[self.iter_idx-1]

    #     return self.train[self.iter_idx-return_size:self.iter_idx]
    
    def build_answer_for_user(
        self,
        data: np.ndarray,
        num_workers: int = 8
    ) -> dict[int, list[int]]:
        users = set(data[:, 0])
        answers: dict[int, list[int]] = dict()
        def process_users(user_ids: np.ndarray) -> None:
            for user_id in user_ids:
                user_id = int(user_id)
                selected_indices = np.where(data[:, 0] == user_id)[0]
                user_data = data[selected_indices]
                # after selected content, sort by weight
                sorted_indices = np.argsort(user_data[:, 2])[::-1]
                user_data = user_data[sorted_indices]
                # get last not interacted item idx
                lowest_weight = user_data[-1, 2]
                first_not_idx = np.greater(user_data[:, 2], lowest_weight).astype(np.int32).sum()
                user_interacted_item_ids = user_data[:first_not_idx, 1].astype(np.int32).tolist()
                # notice that the user_interacted_item_ids may eqal [], ignore
                if len(user_interacted_item_ids) == 0:
                    continue
                answers[user_id] = user_interacted_item_ids
        user_blocks = np.array_split(list(users), num_workers)
        threads = [
            Thread(target=process_users, args=(user_ids,))
            for user_ids in user_blocks
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return answers

    def get_valid_answers(
        self,
        user_ids: list[int] = None
    ) -> list[tuple[int, list[int]]]:
        if user_ids is None:
            return self.valid_answers.items()
        return [ self.valid_answers.get(user_id, []) for user_id in user_ids ]
    
    def get_test_answers(
        self,
        user_ids: list[int]
    ) -> list[list[int]]:
        if user_ids is None:
            return self.test_answers.items()
        return [ self.test_answers.get(user_id, []) for user_id in user_ids ]

    def get_train_answers(
        self,
        user_ids: list[int]
    ) -> list[list[int]]:
        if user_ids is None:
            return self.train_answers.items()
        return [ self.train_answers.get(user_id, []) for user_id in user_ids ]

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
            user_ids: array of id of user for get iteracted items
            ignore_item_ids: each item is the ignore item id for each user

        purpose of ignore item_ids: in training, the pred label may actually in the user iteracted items, so ignore the label
        item as try to let model learn it from other items.
        """
        if pad_item_id is None:
            pad_item_id = self.pad_item_id
        user_iteracted_items = self.get_train_answers(user_ids)

        # remove ignore item ids
        if ignore_item_ids is not None:
            for (interacted_items, ignore_ids) in zip(user_iteracted_items, ignore_item_ids):
                ignore_ids_set = set(ignore_ids)
                interacted_items = [ item_id for item_id in interacted_items if item_id not in ignore_ids_set ]

        # padding
        max_len = max(len(interacted_items) for interacted_items in user_iteracted_items)
        rt_mask = []
        for interacted_items in user_iteracted_items:
            pad_len = max_len - len(interacted_items)
            rt_mask.append([True] * len(interacted_items) + [False] * pad_len)
            interacted_items.extend([pad_item_id] * pad_len)
        
        if return_tensor:
            return Tensor(user_iteracted_items).long(), Tensor(rt_mask).bool()
        return user_iteracted_items, rt_mask
    

class ComposedLoader(KGLoader, UserItemInteractionLoader):

    def __init__(
        self, 
        item_to_entity_id: dict[Any, int], 
        num_entities: int, 
        triplets: list[tuple[int, int, int]], 
        relations: list[str], 
        add_inv: bool,
        users: dict,
        items: dict,
        interactions: np.ndarray,
        portions: tuple[float, float, float] = (0.8, 0.1,)
    ) -> None:
        super(ComposedLoader, self).__init__(item_to_entity_id, num_entities, triplets, relations, add_inv)
        super(KGLoader, self).__init__(users, items, interactions, portions)
        self.item_id2item_idx = { v: i for i, v in enumerate(self.items) }

    @property
    def pad_item_id(self) -> int:
        return self.num_entities - 1

    def convert_entity_ids_to_item_idxes(
        self,
        entity_ids: list[int]
    ) -> list[int]:
        return [ self.item_id2item_idx[self.entity_id2item_id[entity_id]] for entity_id in entity_ids ]

    def convert_item_indexes_to_entity_ids(
        self,
        item_idxes: list[int]
    ) -> list[int]:
        return [ self.item_id2entity_id[self.items[item_idx]] for item_idx in item_idxes ]

    # @TODO: re write get_neighbors method for return entity id not item id
    def read_interacted_items(self, user_ids: list[int], ignore_item_ids: list[list[int]], pad_item_id: Optional[int] = None, return_tensor: bool = True) -> tuple[Tensor, Tensor] | tuple[list[list[int]], list[list[int]]]:
        if pad_item_id is None:
            pad_item_id = self.pad_item_id
        ignore_entity_ids = ignore_item_ids  # actually this is entity id, what fuck
        
        if ignore_entity_ids is not None:
            ignore_item_idxes = [
                self.convert_entity_ids_to_item_idxes(each_user_ignore_ent_ids)
                for each_user_ignore_ent_ids in ignore_entity_ids
            ]
        else:
            ignore_item_idxes = None

        user_iteracted_items_idx, rt_mask = super(KGLoader, self).read_interacted_items(
            user_ids, 
            ignore_item_idxes, 
            -1, 
            False
        )
        
        user_interacted_items = []
        for iteracted_items_idx in user_iteracted_items_idx:
            single_inters = []
            for idx in iteracted_items_idx:
                if idx == -1:
                    single_inters.append(-1)
                else:
                    single_inters.append(self.items[idx])
            user_interacted_items.append(single_inters)
        
        rt_user_interacted_entity_ids = [
            self.convert_item_ids_to_entity_ids(interacted_items, True, pad_item_id)
            for interacted_items in user_interacted_items
        ]
        if return_tensor:
            return Tensor(rt_user_interacted_entity_ids).long(), Tensor(rt_mask).bool()
        return rt_user_interacted_entity_ids, rt_mask
    
    def get_test_set(self):
        item_idxes = self.test[:, 1].tolist()
        entity_ids = self.convert_item_indexes_to_entity_ids(item_idxes)
        test_set = np.stack([self.test[:, 0], entity_ids, self.test[:,1]], -1)
        return test_set
    
    def get_valid_set(self):
        item_idxes = self.valid[:, 1].tolist()
        entity_ids = self.convert_item_indexes_to_entity_ids(item_idxes)
        valid_set = np.stack([self.valid[:, 0], entity_ids, self.valid[:,1]], -1)
        return valid_set
    
    def get_test_answers(self):
        test_answers = self.test_answers
        for user_id in test_answers:
            test_answers[user_id] = self.convert_item_indexes_to_entity_ids(test_answers[user_id])
        return test_answers

    def get_valid_answers(self):
        valid_answers = self.valid_answers
        for user_id in valid_answers:
            valid_answers[user_id] = self.convert_item_indexes_to_entity_ids(valid_answers[user_id])
        return valid_answers

    def __len__(self) -> int:
        return len(self.pos_inters)
    
    def __getitem__(self, index):
        user_idx, item_idx, weight = self.pos_inters[index]
        user_idx, item_idx = int(user_idx), int(item_idx)
        item_id = self.items[item_idx]
        entity_idx = self.item_id2entity_id[item_id]
        # user_id, item_id = super(KGLoader, self).__getitem__(index)
        # entity_id = self.item_to_entity_id[item_id]
        return user_idx, entity_idx
