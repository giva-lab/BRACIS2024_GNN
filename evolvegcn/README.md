EvolveGCN
=====

This repository contains the code for [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), published in AAAI 2020. With some changes for the application in São Paulo city crime data.

In addition there are scripts to handle the data to fit the model necessities.

## Data
 
For the downloaded and processed São Paulo data set, please place the processed data in the 'src/dataloader/data/processed/sp' directory. After that just go to the 'src/dataloader' directory and run the following script:

```sh
python3 egcn_dataloader.py --time-delta M
```
### Command-Line Argument

- `--time_delta`, `-td` (str): Time period to consider when grouping crimes. Options are 'D' (day), 'M' (month), or 'Y' (year). The default is 'M'.

## Output

The script generates several pickle files in the 'src/dataloader/data/dataloader_format_data/sp' directory:

- `nodes_labels_times-<time_delta>.pickle`: Tensor with information about the all the nodes idx, label (0 or 1) and timestep that occurs.
- `edges.pickle`: Dictionary containing information about the edges and timesteps that these edges are.
- `nodes.pickle`: Tensor with information about each node idx and it's features.
- `nodes_feats.pickle`: Tensor similar to nodes.pickle, but whitout idx information.

After that the model can be properly used.

## Requirements
  All the requirements are listed in pyproject.toml file.

## Usage of the model

Go to the 'src/EGCN_model' directory and set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python3 run_sp_exp.py --config_file ./experiments/parameters_example.yaml
```
The configurations for the experiments in our paper are already in the 'src/EGCN_model/experiments' folder by the name 'parameters_sp_egcn_h'.

Most of the parameters in the yaml configuration file are self-explanatory. For hyperparameters tuning, it is possible to set a certain parameter to 'None' and then set a min and max value. Then, each run will pick a random value within the boundaries (for example: 'learning_rate', 'learning_rate_min' and 'learning_rate_max').

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch (Not used for the purposes of our paper, but recommended by the authors of EvolveGCN paper). For example:

```sh
python log_analyzer.py log/filename.log
```

Besides the metrics results, to reproduce the colormap you can go to the "src/color_map" directory, make sure to put in this folder the file "MAP_SOFT_PREDICTIONS.txt" generated in the 'src/EGCN_model' directory after the model finished running and the file "street_nodes_dataframe.pickle" of the processed data. There will be a jupyter notebook to create the map with the help of this two data files, just run all the cells and the html archive will be downloaded in the same folder.

## Reference for the original paper of EvolveGCN

[1] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191). AAAI 2020.

## BibTeX entry for the original paper of EvolveGCN

Please cite their paper if you use this code in your work:

```
@INPROCEEDINGS{egcn,
  AUTHOR = {Aldo Pareja and Giacomo Domeniconi and Jie Chen and Tengfei Ma and Toyotaro Suzumura and Hiroki Kanezashi and Tim Kaler and Tao B. Schardl and Charles E. Leiserson},
  TITLE = {{EvolveGCN}: Evolving Graph Convolutional Networks for Dynamic Graphs},
  BOOKTITLE = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  YEAR = {2020},
}
```
