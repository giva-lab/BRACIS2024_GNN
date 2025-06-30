# Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks (BRACIS 2024)

This repository contains the code and data for our paper titled *"Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks,"* which has been submitted to the BRACIS 2024 conference. The project is organized into three main components:

## 1. Crime Data Processing Pipeline (`crime-data`)
The `crime-data` folder includes a robust pipeline designed to integrate crime data from São Paulo’s Department of Public Safety with urban infrastructure into a graph derived from a street map. This integration provides a detailed and dynamic representation of the crime landscape in São Paulo. The pipeline is versatile and can be applied to various domains and geographical areas, provided the data is geolocated.

## 2. Dynamic Self-Attention Network (DySAT) (`dysat`)
The `dysat` folder implements the Dynamic Self-Attention Network (DySAT), an unsupervised graph embedding model that learns latent node representations to capture dynamic graph structures. DySAT employs self-attention mechanisms across both structural and temporal dimensions to generate accurate and robust node embeddings. The model's architecture includes:

- **Spatial Self-Attention:** Captures structural associations in each snapshot of the dynamic graph.
- **Temporal Self-Attention:** Learns temporal dependencies to understand how node representations evolve over time.
- **Output Layer:** Integrates spatial and temporal embeddings for tasks like node classification, link prediction, and anomaly detection.

## 3. Evolving Graph Convolutional Networks (EvolveGCN) (`evolvegcn`)
The `evolvegcn` folder contains the implementation of Evolving Graph Convolutional Networks (EvolveGCN), which adapt Graph Convolutional Networks (GCNs) to handle continuously evolving graphs. EvolveGCN uses time-series models, specifically LSTM networks, to predict and adjust GCN parameters over time, ensuring the network adapts to the graph's changing dynamics. This approach is essential for real-world applications where relationships between entities evolve, such as in social networks or communication networks.

## Usage
Each folder contains detailed instructions on setting up the environment, preparing the data, and running the models. Please refer to the respective `README` files within each folder for more specific guidelines.

## Citation
If you use this code, please cite our paper:
```
@inproceedings{hassan2024modeling,
  title={Modeling and Predicting Crimes in the City of S{\~a}o Paulo Using Graph Neural Networks},
  author={Hassan, Waqar and Cabral, Marvin Mendes and Ramos, Thiago Rodrigo and Filho, Antonio Castelo and Nonato, Luis Gustavo},
  booktitle={Brazilian Conference on Intelligent Systems},
  pages={372--386},
  year={2024},
  organization={Springer}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
