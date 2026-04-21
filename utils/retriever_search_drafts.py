import json
import torch
import numpy as np
import networkx as nx

from collections import deque
from utils.contriever import Retriever


def edge(text):
    """Parse triplet string 'subject, relation, object'. Handles commas in subject/object."""
    parts = [e.strip() for e in text.split(",")]
    if len(parts) < 3:
        return parts  # Let caller handle invalid format
    # Subject, relation, object - object may contain commas
    return [parts[0], parts[1], ",".join(parts[2:])]


def build_graph(triplets):
    G = nx.Graph()
    for t in triplets:
        parts = edge(t)
        if len(parts) < 3:
            continue
        src, e, trg = parts[0], parts[1], parts[2]
        G.add_edge(src, trg, relation=e)

    return G


def add_similar(G, src, visited, query, dist, retriever, threshold=1.5):
    result = retriever.search(
        list(G.nodes),
        src,
        similarity_threshold=threshold,
        return_scores=True
    )
    similar = [s for s in result['strings'] if s not in visited]
    visited.update(similar)
    query.extend(similar)
    for s in similar:
        dist[s] = dist[src]

@torch.no_grad()
def claster(nodes, retriever: Retriever):
    curr_nodes = nodes
    print('Initial nodes:\n', " | ".join(curr_nodes), sep="")
    np.set_printoptions(precision=3)
    while True:
        print('=========================')
        embeds = retriever.embed(curr_nodes)
        scores = embeds @ embeds.T
        scores.fill_diagonal_(-1.)
        scores = scores.numpy()
        mmax = scores.max()
        if mmax <= 1.:
            break
        #print(f'Scores ({pair_scores.shape}):\n', pair_scores)
        x,y = np.unravel_index(scores.argmax(), scores.shape)
        combo_node = ", ".join([curr_nodes[x], curr_nodes[y]])
        print(f"Join [{curr_nodes[x]}] and [{curr_nodes[y]}], with similarity score={mmax:.3f}")
        curr_nodes = [n for i, n in enumerate(curr_nodes) if i not in (x, y)]
        curr_nodes.append(combo_node)
        #print("New nodes:\n", " | ".join(curr_nodes), sep="")

    print("Final nodes:\n", "\n".join(curr_nodes), sep="")

def graph_retr_search_old(
        start_query,
        triplets,
        retriever: Retriever,
        max_depth :int=2,
        topk :int=3,
        post_retrieve_threshold: float=0.7, #not exclusive with topk here
        verbose=2,
):

    queue = deque()
    queue.append(start_query)
    d = {start_query: 0}

    result = []
    print(f"queue : {queue}")
    while queue:
        q = queue.popleft()
        if d[q] >= max_depth: continue
        print(f"Searching {len(triplets)} triplets for query {q}")
        res = retriever.search(triplets, q, topk=topk, return_scores=True)
        print(f"Found {len(res['strings'])} triplets")
        for s, score in zip(res['strings'], res['scores']):
            if score < post_retrieve_threshold: continue
            print(f"triplet {s} with score {score}")
            parts = edge(s)
            if len(parts) < 3:
                print(f"malformed triplet {s}")
                continue  # Skip malformed triplet strings
            v1, e, v2 = parts[0], parts[1], parts[2]
            for v in [v1, v2]:
                if v not in d:
                    queue.append(v)
                    print(f"adding {v} to queue")
                    d[v] = d[q] + 1
            if s not in result:
                result.append(s)
                print(f"adding {s} to result")
    return result


def graph_retr_search(
        queries,
        triplets,
        retriever: Retriever,
        topk :int=3,
        post_retrieve_threshold: float=0.7, #not exclusive with topk here
        verbose=2,
        #flatten_results=False,
):

    
    result = []

    #print(f"Searching {len(triplets)} triplets for {len(queries)} queries")
    res = retriever.search(triplets, queries, topk=topk, return_scores=True#, flatten_results=flatten_results
    ) # dict with values = 2d lists
    #print(f"Found {len(res['strings'])} answers")

    for strings, scores in zip(res['strings'], res['scores']):
        query_result = []
        for s, score in zip(strings, scores):
            if score < post_retrieve_threshold: continue
            #print(f"triplet {s} with score {score}")
            parts = edge(s)
            if len(parts) < 3:
                #print(f"malformed triplet {s}")
                continue  # Skip malformed triplet strings
            if s not in result:
                query_result.append(s)
                #print(f"adding {s} to result")
        result.append(query_result)
    return result

def eval_triplets(triplets):
    reference_full = [
        'recipe #1, instructs, prepare meal',
        'recipe #1, requires, orange bell pepper',
        'orange bell pepper, to be, diced', 'orange bell pepper, to be, grilled', 'bbq, used for, grilling',
        'fridge, contains, orange bell pepper', 'kitchen, contains, fridge',
        'recipe #1, requires, green bell pepper', 'green bell pepper, to be, diced', 'green bell pepper, to be, fried',
        'stove, used for, frying', 'green bell pepper, is in, garden',
        'recipe #1, requires, yellow potato', 'yellow potato, to be, sliced', 'yellow potato, to be, grilled',
        'bbq, used for, grilling', 'yellow potato, is in, garden'
    ]
    is_found = [int(r in triplets) for r in reference_full]
    print(f"Found {sum(is_found)}/{len(is_found)} from reference_full")
    
    
def graph_retr_search_thesises(
        start_query,
        thesises, entities,
        retriever: Retriever,
        max_depth :int=2,
        topk :int=3,
        post_retrieve_threshold: float=0.7, #not exclusive with topk here
        verbose=2,
):

    queue = deque()
    queue.append(start_query)
    d = {start_query:0}

    result = []

    while queue:
        q = queue.popleft()
        if d[q] >= max_depth: continue

        only_names = [thesis.name for thesis in thesises.values()]
        list_of_ids = [key for key in thesises.keys()]
        res = retriever.search(only_names, q, topk=topk, return_scores=True)
        for i, score in zip(res['idx'], res['scores']):
            if score < post_retrieve_threshold: continue
            for v_id in thesises[list_of_ids[i]].children:
                v = entities[v_id].name
                if v not in d:
                    queue.append(v)
                    d[v] = d[q] + 1
            if thesises[list_of_ids[i]].name not in result:
                result.append(thesises[list_of_ids[i]].name)

    return result

   