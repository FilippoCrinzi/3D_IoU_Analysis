


# Shape Estimation of Vehicles from Point Clouds
This project implements a framework for quantitatively evaluating the geometric reconstruction of vehicles from point cloud data. The objective is to assess the performance of different parametric shapes: such as bounding boxes, ellipsoids, and superquadrics in approximating the geometry of detected vehicles in simulated environments.

The evaluation compares the reconstructed shapes against ground truth 3D models using metrics like Intersection over Union (IoU), focusing on aspects such as shape accuracy, orientation, and spatial positioning. The goal is to contribute to the development of more reliable and precise perception systems, particularly for applications in autonomous driving and robotics.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modules](#modules)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/FilippoCrinzi/3D_IoU_Analysis.git
    cd 3D_IoU_Analysis
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script:
    ```sh
    python main.py
    ```

2. Follow the command-line instructions.

## Project Structure
```plaintext
3D_IoU_Analysis/
├── data/
    └──models/
    └──pointclouds/
    └──sensor_position/
    └──trajectories/
├── results/
    └──compare_graph.png/
    └──compare_results.csv/
    └──epsilon_analysis.csv/
    └──epsilon_summary.csv/
    └──IoU_align_graph.png/
    └──IoU_graph.png/
    └──IoU_results.csv/
    └──trade_off.png/
    └──trade_off_results.csv/
    └──trade_off_time.csv/
    └──trade_off_time.png/
├── src/
    └──intersection_over_union.py/
    └──main.py/
    └──point_sampling.py/
    └──shape_generator.py/
    └──trade_off.py/
    └──utils.py/
├── README.md
├── requirements.txt
```
## Modules
- **main.py**: Main script to run the evaluation framework.
- **intersection_over_union.py**: Module to compute the Intersection over Union (IoU) metric.
- **point_sampling.py**: Module to sample points from 3D models.
- **shape_generator.py**: Module to generate parametric shapes (bounding boxes, ellipsoids, superquadrics).
- **trade_off.py**: Module to analyze the trade-off between IoU precision and efficiency.
- **utils.py**: Utility functions for data loading, plotting, and evaluation.

 