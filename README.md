# Embedding Guided Layout

Recommended Python version is 3.7.6.
Packages are listed in ``requirements.txt``.

Install requirements:
```Bash
pip install -r requirements
```

Run tests:
```Bash
python3 main.py
```

Results are stored in ``./fig/<graphname>/<datetime>``.

``overall_graphs.json`` is a json configuration file for overall test：

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
