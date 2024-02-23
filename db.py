from pathlib import Path
from typing import Optional, Union

import torch
from neomodel import db, config, StructuredNode, IntegerProperty, StringProperty, \
    StructuredRel, RelationshipTo, RelationshipManager, RelationshipFrom
import numpy as np
import pandas as pd
from torch import Tensor

from model import Iloader


def override(func):
    r""" declare functions is override from super """
    return func

class MovieToNotMovieRel(StructuredRel):
    name = StringProperty(required=True, unique_index=True)
    relation_id = IntegerProperty(unique_index=True)


class MovieToMovieRel(StructuredRel):
    name = StringProperty(required=True, unique_index=True)
    relation_id = IntegerProperty(unique_index=True)


class Rate(StructuredRel):
    rate = IntegerProperty(required=True)


class Movie(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    item_id = IntegerProperty(required=True, unique_index=True) # in ml-20m id
    entity_id = IntegerProperty(required=True, unique_index=True) # in kg id
    tags = StringProperty()

    rel_to_not_movie = RelationshipTo('NotMovieNode', 'REL_TO_NOT_MOVIE', model=MovieToNotMovieRel)
    rel_to_movie = RelationshipTo('Movie', 'REL_TO_MOVIE', model=MovieToMovieRel)


class NotMovieNode(StructuredNode):
    entity_id = IntegerProperty(required=True, unique_index=True)

    rel_to_movie = RelationshipTo('Movie', 'REL_TO_MOVIE', model=MovieToMovieRel)
    rel_to_not_movie = RelationshipTo('NotMovieNode', 'REL_TO_NOT_MOVIE', model=MovieToNotMovieRel)


class User(StructuredNode):
    entity_id = IntegerProperty(required=True, unique_index=True)

    rate = RelationshipTo('Movie', 'RATE', model=Rate)

# for artists db
    

class ArtistRelation(StructuredRel):
    relation_id = IntegerProperty(required=True,unique_index=True)
    name = StringProperty(unique_index=True)


class ArtistEntity(StructuredNode):
    entity_id = IntegerProperty(required=True, unique_index=True)
    item_id = IntegerProperty(required=False, unique_index=True)

    rel_to = RelationshipTo('ArtistEntity', ArtistRelation.__name__, model=ArtistRelation)
    rel_from = RelationshipFrom('ArtistEntity', ArtistRelation.__name__, model=ArtistRelation)


def connect_to_neo4j(address: str, username: str, password: str):
    r"""
    address is like: 'localhost:7687'
    """
    config.DATABASE_URL = f'bolt://{ username }:{ password }@{ address }'
    db.set_connection(f'bolt://{ username }:{ password }@{ address }')
	

@db.transaction
def build_neo4j_movie_dataset(movies, triplets, relations, clear=True):
    if clear:
        db.cypher_query("MATCH (n) DETACH DELETE n")

    Movie.create(*movies)

    for triplet in triplets:
        s, r, o = triplet
        r"""
        cases:
        1. s is movie, o is movie
        2. s is movie, o is not movie
        3. s is not movie, o is movie
        4. s is not movie, o is not movie
        if s is movie, then s < len(movies) or s != any_movie.entity_id
        for safety we use the next inequation
        """
        s_node: Optional[Movie] = Movie.nodes.get_or_none(entity_id=s)
        o_node: Optional[Movie] = Movie.nodes.get_or_none(entity_id=o)
        if o < 16954:
            print(f"actualy o {o} is movie")
        if s_node is not None and o_node is not None:  # case 1
            s_node.rel_to_movie.connect(o_node, {'name': relations[r]})
        elif s_node is not None and o_node is None:  # case 2
            o_node = NotMovieNode(entity_id=o).save()
            s_node.rel_to_not_movie.connect(o_node, {'name': relations[r]})
        elif s_node is None and o_node is not None:  # case 3
            s_node = NotMovieNode(entity_id=s).save()
            s_node.rel_to_movie.connect(o_node, {'name': relations[r]})
        else:
            s_node = NotMovieNode(entity_id=s).save()
            o_node = NotMovieNode(entity_id=o).save()
            s_node.rel_to_not_movie.connect(o_node, {'name': relations[r]})


@db.transaction
def build_neo4j_artist_dataset(entities: list[dict], triplets: list[int], relations: list[str], clear=True):
    if clear:
        db.cypher_query("MATCH (n) DETACH DELETE n")
    
    ArtistEntity.create(*entities)
    for s, r, o in triplets:
        s_node: Optional[ArtistEntity] = ArtistEntity.nodes.get_or_none(entity_id=s)
        o_node: Optional[ArtistEntity] = ArtistEntity.nodes.get_or_none(entity_id=o)
        if s_node is None:
            s_node = ArtistEntity(entity_id=s).save()
        if o_node is None:
            o_node = ArtistEntity(entity_id=o).save()
        s_node.rel_to.connect(o_node, {'name': relations[r], 'relation_id': r})
        o_node.rel_from.connect(s_node, {'name': relations[r], 'relation_id': r})
        

class DbLoader(Iloader):

    def __init__(
        self, 
        item_node_cls: type[StructuredNode],
        relation_names: list[str],
        num_paths: int,
        num_steps: int,
        num_collect: int
    ) -> None:
        self.item_node_cls = item_node_cls
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.relation_names = relation_names
        self.num_collect = num_collect

    @override
    def get_neighbors(
            self, 
            entity_ids: list[int], 
            return_tensor: bool = True
        ) -> Union[tuple[Tensor, Tensor, Tensor], tuple[list[int], list[int], list[int]]]:
        r"""
        Args:
            entity_ids: entities needs to get the neighbours.
        Returns:
            0 neighbour_ids: shape(num_entities, num_neighbours)
            1 relation_ids: shape(num_entities, num_neighbours)
            2 neighbour_masks: True for masked, shape(num_entities, num_neighbours)
        """
        neighbours = []
        relations = []
        masks = []

        max_num_neighbours = 0
        for entity_id in entity_ids:
            item = self.item_node_cls.nodes.get_or_none(entity_id=entity_id)
            if item is None:
                raise ValueError(f'entity id {entity_id} not found in db')
            neighbour_ids = []
            relation_ids = []
            for rel_name in self.relation_names:
                rel: RelationshipManager = getattr(item, rel_name)
                nei_nodes = rel.all()

                neighbour_ids.extend([ n.entity_id for n in nei_nodes ])
                relation_ids.extend([ rel.relationship(n).relation_id for n in nei_nodes ])
            
            if len(neighbour_ids) > max_num_neighbours:
                max_num_neighbours = len(neighbour_ids)
            neighbours.append(neighbour_ids)
            relations.append(relation_ids)
        
        # padding
        for i in range(len(neighbours)):
            neighbours[i] = neighbours[i] + [0] * (max_num_neighbours - len(neighbours[i]))
            relations[i] = relations[i] + [0] * (max_num_neighbours - len(relations[i]))
            masks.append([0] * len(neighbours[i]) + [1] * (max_num_neighbours - len(neighbours[i])))
        
        if return_tensor:
            return (
                torch.tensor(neighbours, dtype=torch.long),
                torch.tensor(relations, dtype=torch.long),
                torch.tensor(masks, dtype=torch.long)
            )
        return neighbours, relations, masks

