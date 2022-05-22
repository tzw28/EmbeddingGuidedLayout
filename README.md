# Embedding Guided Layout

Recommended Python version is 3.7.6.
Packages are listed in ``requirements.txt``.

+ Install requirements:
```Bash
pip install -r requirements
```

+ Run node2vec-a embedding:

Given a Networkx graph G, run node2vec-a to get embedding results:

```python
from src.attr2vec.embedding import attributed_embeddingfrom
from src.util.graph_reading import init_attributed_graph, clean_attributed_graph

virtual_nodes = init_attributed_graph(G)
vectors, walks = attributed_embedding(
    G,                                  # Networkx graph
    return_weight=return_weight,        # 1/p
    neighbor_weight=neighbor_weight,    # 1/q
    attribute_weight=attribute_weight,  # 1/r
    virtual_nodes=virtual_nodes         # Extra virtual node list
)
clean_attributed_graph(G, vectors)
```

Embedding vectors are stored in dict `vectors`.

+ Run GEGraph layout:
```python
from src.attr2vec.embedding import attributed_embeddingfrom src.from util.graph_reading import init_attributed_graph, clean_attributed_graph

pos = embedding_fr(
    G,                  # Networkx graph
    vectors=vectors,    # Embedding vectors
    wa=wa,              # Weight of adjacent matrix
    we=1-wa,            # Weight of similarity matrix
    cluster=node_class, # Node class dict, {"%node%": "%class%"}
    tel=tel,            # Intra-community truncation threshold
    teh=teh             # Inter-community truncation threshold
)
```

Positions of node are stored in dict `pos`.

+ Generate layout-preserving-aggregation view:
```python
from src.util.key_words import graph_tf_idf
from src.aggregation.graph_aggregation import GraphAggregator

weights = graph_tf_idf(
    G,                                  # Networkx graph
    cluster,                            # Clustering result or node class
    walks,                              # Random walk paths. Used as context.
)

agg = GraphAggregator(
    G,                          # Networkx graph
    pos,                        # Node positions
    group_attribute="group",    # The community attribute of each node
    fig_size=10,                # View port width and height
    ax_gap=0.05,                # Gap between view port boundary and layout axis
    is_curved=True,             # Draw Bezier curves for edges linking two communities
    attr_vectors=vectors,       # Input embedding vectors
    weights=weights
)
agg.generate_aggregations()     # Precompute
agg.draw_aggregations(          # Visualize
    size_min=agg_size_min,      # Minimum size of aggregation node 
    size_max=agg_size_max,      # Maximum size of aggregation node 
    width_min=agg_width_min,    # Minimum width of edges betweent aggregation nodes
    width_max=agg_width_max,    # Maximum width of edges betweent aggregation nodes
    agg_alpha=agg_alpha,        # Transparency of the background of clicked aggregation node
)
agg.set_events(fig)
```
When the Matplotlib window shows, click any aggregation node to show the detail of the community.

+ Generate related-nodes-searching view
```python
from src.util.plot_drawing import draw_embedding_fr

draw_embedding_fr(
    graph_file=graph_name,              # Graph name
    default_pos=pos,                    # Percomputed positions
    attribute_weight=attribute_weight,  # 1/r
    neighbor_weight=neighbor_weight,    # 1/q
    return_weight=return_weight,        # 1/p
    seed=seed,                          # Seed for random
    color=cluster,                      # Communities for node color
    size_list=node_size,                # Node sizes
    width=edge_width,                   # Edge width
    edge_alpha=edge_alpha,              # Edge alpha
    wa=wa,                              # Weight of adjacent matrix
    we=we,                              # Weight of similarity matrix
    tel=tel,                            # Intra-community truncation threshold
    teh=teh,                            # Inter-community truncation threshold
    fig=fig,                            # Matplotlib figure object
    ax=ax,                              # Matplotlib axis object
    ax_gap=0.05,                        # Gap between view port boundary and layout axis
    fig_size=10,                        # View port width and height
    click_on=True,                      # Enable node clicking
)
```

Single click a node in the Matplotlib window, the similar nodes will be emphasized with different colors. Click the same node again, the nodes with attribute, local and global proximity will be shown recurrently. 

+ Run batch tests:

```python
from tests.overall_test import run_overall_tests

run_overall_tests()
```

Drawn graphs are stored in ``./fig/<graphname>/<datetime>``.

``overall_graphs.json`` is a json configuration file for each graph in overall test：

```
{
    "miserables": {               // Graph name
        "note": "overall",        // Description of test
        "seed": 6,                // Seed for random
        "k": 8,                   // K-Means: Cluster number
        "d": 8,                   // node2vec-a: Dimension of vector
        "walklen": 30,            // node2vec-a: Length of random walk path
        "attribute_weight": 0.6,  // node2vec-a: 1/r
        "neighbor_weight": 4,     // node2vec-a: 1/q
        "return_weight": 1,       // node2vec-a: 1/p
        "epochs": 15,             // node2vec-a: Paths starting from each node
        "tel": 0.4,               // GEGraph: Truncation parameter for nodes in the same groups
        "teh": 0.6,               // GEGraph: Truncation parameter for nodes in different groups
        "wa": 0.4,                // GEGraph: Weight of adjacency matrix
        "we": 0.6,                // GEGraph: Weight of similarity matrix
        "node_size": 30,          // Drawing: Node size
        "edge_width": 0.2,        // Drawing: Edge width
        "edge_alpha": 0.2,        // Drawing: Edge alpha
        "agg_size_min": 4000,     // Application: The minimum size of the aggregated groups
        "agg_size_max": 12000,    // Application: The maximum size of the aggregated groups
        "agg_width_min": 6,       // Application: The minimum size of the links between the aggregated groups
        "agg_width_max": 40,      // Application: The maximum size of the links between the aggregated groups
        "agg_alpha": 0.1,         // Application: The alpha of the links between the aggregated groups
    },
    ...
}
```
