import argparse
import pathlib
import pickle

import networkx as nx
import numpy as np

# THIS SCRIPT IS ONLY FOR THE APPLICATION PURPOSE

parser = argparse.ArgumentParser()


parser.add_argument(
    "--time_delta",
    "-td",
    default="M",
    choices=["D", "M", "Y"],
    type=str,
    help="Time period to consider when grouping crimes in (Y)ear, (D)ay or (M)onth. Default is (M)onth.",
)

args = parser.parse_args()


processed_data_path = pathlib.Path(f"./data/processed/sp")

output_path = pathlib.Path("./data/dataloader_format_data/sp/")
output_path.mkdir(exist_ok=True, parents=True)

assert (
    processed_data_path.is_dir()
), "Invalid processed data. Did you put the processed data in the /src/dataloader/data/processed/sp folder?"


with open(processed_data_path / "crime_nodes_dataframe.pickle", "rb") as f:
    crime_df = pickle.load(f)

with open(processed_data_path / "street_nodes_dataframe.pickle", "rb") as f:
    nodes_df = pickle.load(f)

with open(processed_data_path / "gdf_edges.pickle", "rb") as f:
    edges_df = pickle.load(f)

with open(processed_data_path / "graph-streets_as_nodes.pickle", "rb") as f:
    graph_nx = pickle.load(f)

# Processing table containing nodes connections (edges) given time

adjacency_matrix = nx.adjacency_matrix(graph_nx).toarray()

edges = np.array([[None, None, None]])

for i in range(adjacency_matrix.shape[0]):
    row = adjacency_matrix[i]
    non_zero = row.nonzero()
    padded = np.pad(non_zero, ((0, 1), (0, 0)), mode="constant", constant_values=i)
    padded = np.pad(padded, ((0, 1), (0, 0)), mode="constant", constant_values=0)
    edge_i = padded.transpose()
    edge_i = edge_i[:, [1, 0, 2]]

    edges = np.concatenate((edges, edge_i))

edges = edges[1:,]

dict_edges = {"idx": edges, "vals": np.array([1] * edges.shape[0])}

with open(output_path / "edges.pickle", "wb") as f:
    pickle.dump(dict_edges, f)


# Processing table containing nodes and label, given time

crime_df = crime_df.drop(
    columns=[
        "latitude",
        "longitude",
        "geometry",
        "crime_coord",
    ]
)

crime_df["day"] = crime_df["time"].dt.day
crime_df["month"] = crime_df["time"].dt.month
crime_df["year"] = crime_df["time"].dt.year

n_nodes = adjacency_matrix.shape[0]

time_step = 0

nodes_labels_times = np.array([[None, None, None]])

for t in crime_df["time"].dt.to_period(args.time_delta).unique():
    nodes_labels = np.zeros((n_nodes, 3), dtype=int)

    nodes_labels[:, 0] = np.arange(n_nodes)
    nodes_labels[:, 2] = time_step

    crime_time = crime_df[crime_df["time"].dt.to_period(args.time_delta) == t]

    nodes_where_crime = crime_time["index_right"].unique()

    nodes_labels[nodes_where_crime, 1] += 1

    time_step += 1

    nodes_labels_times = np.concatenate([nodes_labels_times, nodes_labels])

nodes_labels_times = nodes_labels_times[1:]

with open(output_path / f"nodes_labels_times-{args.time_delta}.pickle", "wb") as f:
    pickle.dump(nodes_labels_times, f)

# Processing nodes features

nodes_feats = edges_df.drop(
    columns=[
        "geometry",
        "street_segment_31983",
        "street_segment",
        "highway",
        "distance_tobarueri",
        "distance_tointerlagos",
        "distance_tomirante",
    ]
)

nodes = np.array(nodes_feats.reset_index())
nodes_feats = np.array(nodes_feats)

with open(output_path / "nodes.pickle", "wb") as f:
    pickle.dump(nodes, f)

with open(output_path / "nodes_feats.pickle", "wb") as f:
    pickle.dump(nodes_feats, f)
