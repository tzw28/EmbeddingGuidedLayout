import random
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from tqdm import tqdm


def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
                            first_travel_key=None):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()
    # print('parallel_generate_walks')
    with tqdm(total=num_walks) as pbar:
        pbar.set_description('Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):
            # print('cpu_num n_walk', cpu_num, n_walk)
            # Update progress bar
            pbar.update(1)

            # Shuffle the nodes
            shuffled_nodes = list(d_graph.keys())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:

                # Skip nodes with specific num_walks
                if source in sampling_strategy and \
                        num_walks_key in sampling_strategy[source] and \
                        sampling_strategy[source][num_walks_key] <= n_walk:
                    continue

                # Start walk
                walk = [source]

                # Calculate walk length
                if source in sampling_strategy:
                    walk_length = sampling_strategy[source].get(
                        walk_length_key, global_walk_length)
                else:
                    walk_length = global_walk_length

                # Perform walk
                while len(walk) < walk_length:

                    walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                    # Skip dead end nodes
                    if not walk_options:
                        break

                    if len(walk) == 1:  # For the first step
                        probabilities = d_graph[walk[-1]][first_travel_key]
                        walk_to = np.random.choice(
                            walk_options, size=1, p=probabilities)[0]
                    else:
                        probabilities = d_graph[walk[-1]
                                                ][probabilities_key][walk[-2]]
                        walk_to = np.random.choice(
                            walk_options, size=1, p=probabilities)[0]

                    walk.append(walk_to)

                walk = list(map(str, walk))  # Convert all to strings

                walks.append(walk)

            # print('cpu_num n_walk end', cpu_num, n_walk)
    # print('parallel_generate_walks end')
    return walks


class Attri2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'
    R_KEY = 'r'

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, r=1, weight_key='weight',
                 workers=1, sampling_strategy=None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param r: Attribute parameter (default: 1）
        :type r: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.r = r
        self.weight_key = weight_key
        self.workers = workers

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.d_graph = self._precompute_probabilities()
        # print('after _precompute_probabilities')
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """
        d_graph = defaultdict(dict)
        first_travel_done = set()

        # print('_precompute_probabilities')

        for source in tqdm(self.graph.nodes(), desc='Computing transition probabilities'):
            # print('_precompute_probabilities', source)
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY, self.p)\
                        if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY, self.q)\
                        if current_node in self.sampling_strategy else self.q
                    r = self.sampling_strategy[current_node].get(self.R_KEY, self.r)\
                        if current_node in self.sampling_strategy else self.r

                    # if destination == source:  # Backwards probability
                    #     ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    # elif destination in self.graph[source]:  # If the neighbor is connected to the source
                    #     ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    # else:
                    #     ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    cns = str(current_node)
                    dns = str(destination)
                    # if cns.startswith('attri-') or dns.startswith('attri-'):
                    # if dns.startswith('attri-'):
                    # if cns.startswith('attri-'):
                    # if cns.startswith('attri-') or dns.startswith('attri-'):
                    if False:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / r
                    else:
                        if destination == source:  # Backwards probability
                            ss_weight = self.graph[current_node][destination].get(
                                self.weight_key, 1) * 1 / p
                        # If the neighbor is connected to the source
                        elif destination in self.graph[source]:
                            ss_weight = self.graph[current_node][destination].get(
                                self.weight_key, 1)
                        else:
                            ss_weight = self.graph[current_node][destination].get(
                                self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(
                            self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / \
                        unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors
        # print('_precompute_probabilities end')
        return d_graph

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """
        # print('sublist')
        def flatten(l): return [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        # print('num_walks_lists', num_walks_lists)

        walk_results = Parallel(n_jobs=self.workers)(delayed(parallel_generate_walks)(self.d_graph,
                                                                                      self.walk_length,
                                                                                      len(num_walks),
                                                                                      idx,
                                                                                      self.sampling_strategy,
                                                                                      self.NUM_WALKS_KEY,
                                                                                      self.WALK_LENGTH_KEY,
                                                                                      self.NEIGHBORS_KEY,
                                                                                      self.PROBABILITIES_KEY,
                                                                                      self.FIRST_TRAVEL_KEY) for
                                                     idx, num_walks
                                                     in enumerate(num_walks_lists, 1))
        # print('walk_results', walk_results)
        walks = flatten(walk_results)
        # print('walks', walks)
        return walks

    def fit(self, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return Word2Vec(self.walks, **skip_gram_params)
