# Graph Neural Networks for Survival Prediction

## Overview
This project implements a Graph Neural Network (GNN) to predict survival outcomes using the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic). The goal is to leverage the relational structure of passenger data (e.g., family relationships, ticket groups) to improve survival prediction accuracy compared to traditional machine learning models. The project is implemented in Python using PyTorch Geometric for GNN modeling and is based on the Kaggle notebook by Shubhranshu331.

## Why This Project?
The project was chosen for the following reasons:
- **Innovative Approach**: GNNs are a powerful tool for modeling relational data, which is underutilized in survival prediction tasks like the Titanic challenge. This project explores how graph-based methods can outperform conventional models.
- **Real-World Relevance**: Survival prediction is critical in fields like healthcare, disaster response, and risk assessment. The Titanic dataset, while historical, serves as a well-structured benchmark for testing such models.
- **Learning Opportunity**: The project provides hands-on experience with GNNs, PyTorch Geometric, and advanced data preprocessing, making it an excellent learning resource for graph-based machine learning.
- **Community Engagement**: The Titanic dataset is a popular Kaggle competition, allowing for comparison with other models and contributions to the data science community.

## What Is This Project Doing?
This project constructs a graph representation of the Titanic dataset, where:
- **Nodes** represent passengers, with features like age, sex, fare, and class.
- **Edges** represent relationships, such as family ties (siblings, parents) or shared tickets, capturing social and group dynamics.
- A **Graph Neural Network** is trained to predict whether a passenger survived (1) or not (0) based on these features and relationships.

### Key Steps:
1. **Data Preprocessing**:
   - Load and clean the Titanic dataset (train and test sets).
   - Extract features (e.g., age, sex, Pclass) and create relational edges (e.g., family, ticket groups).
   - Convert data into a graph format compatible with PyTorch Geometric.
2. **Graph Construction**:
   - Nodes are passengers with feature vectors.
   - Edges connect passengers based on family relationships or shared tickets.
3. **GNN Model**:
   - A GNN model (using Graph Convolutional Networks or similar layers) aggregates information from neighboring nodes to predict survival.
   - The model is trained with a binary cross-entropy loss and optimized using Adam.
4. **Evaluation**:
   - Performance is evaluated using accuracy, ROC-AUC, and a ROC curve visualization.
   - Predictions are generated for the test set and saved for Kaggle submission.

## How Does This Project Work?
The project workflow is as follows:
1. **Dependencies**:
   - Python 3.11, PyTorch, PyTorch Geometric, Pandas, NumPy, Scikit-learn, Matplotlib.
   - GPU acceleration (e.g., NVIDIA Tesla T4) is supported for faster training.
2. **Data Preparation**:
   - The Titanic dataset is loaded from Kaggle (train.csv, test.csv).
   - Missing values are imputed (e.g., median age, mode for embarked).
   - Features are engineered (e.g., family size, title from names).
   - A graph is built using PyTorch Geometric, with edges based on family or ticket relationships.
3. **Model Training**:
   - A GNN model is defined with multiple graph convolutional layers.
   - The model is trained on the training set, with node features and edge connections as input.
   - Hyperparameters (e.g., learning rate, number of epochs) are tuned for optimal performance.
4. **Evaluation and Visualization**:
   - The model is evaluated on a validation split using accuracy and ROC-AUC.
   - A ROC curve is plotted and saved as `roc_curve.png`.
   - Predictions are generated for the test set and saved as a CSV file for Kaggle submission.
5. **Execution**:
   - The code is structured in a Jupyter Notebook (`Graph_Neural_Networks_for_Survival_Prediction.ipynb`).
   - Run the notebook in a Kaggle environment or locally with the required dependencies and dataset.

## What Is the Use of This Project?
- **Survival Prediction**: The model predicts survival outcomes, which can be applied to similar tasks in medical prognosis, disaster response, or risk analysis.
- **Graph-Based Insights**: By modeling relationships, the GNN captures group dynamics (e.g., families surviving together), offering insights traditional models might miss.
- **Benchmarking**: The project serves as a benchmark for comparing GNNs with other machine learning models on the Titanic dataset.
- **Educational Tool**: It demonstrates how to preprocess data, build graphs, and implement GNNs, making it valuable for students and researchers learning graph-based ML.
- **Kaggle Contribution**: The project contributes to the Kaggle community by providing a novel approach to a classic problem, potentially improving leaderboard scores.

## Relevance of the Project
- **Advancing ML Techniques**: GNNs are increasingly relevant in domains with relational data (e.g., social networks, molecular chemistry, recommendation systems). This project showcases their application in survival prediction.
- **Interdisciplinary Impact**: Survival prediction models have applications in healthcare (e.g., patient survival), transportation (e.g., safety analysis), and sociology (e.g., group behavior).
- **Scalability**: The approach can be extended to larger datasets or other graph-based prediction tasks, making it versatile.
- **Community and Research**: By sharing the code on GitHub and Kaggle, the project encourages collaboration, reproducibility, and further research into GNNs for tabular and relational data.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/graph-neural-networks-survival-prediction.git
   cd graph-neural-networks-survival-prediction
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure PyTorch Geometric is installed (see [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)).
3. **Download the Dataset**:
   - Get the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data).
   - Place `train.csv` and `test.csv` in the `data/` folder.
4. **Run the Notebook**:
   - Open `Graph_Neural_Networks_for_Survival_Prediction.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute all cells to preprocess data, train the model, and generate predictions.
5. **View Results**:
   - Check `roc_curve.png` for the ROC curve.
   - Find test predictions in the generated `submission.csv`.

## Requirements
- Python 3.11
- PyTorch
- PyTorch Geometric
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Kaggle API (for dataset access)

## Results
- **Accuracy**: Competitive accuracy on the Titanic test set (varies based on hyperparameter tuning).
- **ROC-AUC**: Strong ROC-AUC score, visualized in `roc_curve.png`.
- **Kaggle Submission**: The model generates predictions suitable for Kaggle submission, potentially ranking well on the leaderboard.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## Acknowledgments
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) for the dataset.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN tools.
- Shubhranshu331 for the original Kaggle notebook inspiration.
