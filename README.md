# Forward and Backward Feature Selection

This repository contains scripts and Jupyter Notebook examples for performing forward and backward feature selection (Greedy).

## Table of Contents

- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Usage](#usage)

## Introduction

Forward and backward feature selection are powerful techniques for identifying the optimal set of features in your project. It's important to note that, due to the nature of the greedy algorithm employed, this process involves exploring all combinations of features, resulting in a time complexity of O(n^2).

While this exhaustive search may take some time, especially with a large number of features, it is a thorough approach that considers various feature combinations. It is recommended to utilize this feature selection method in the final stages of your selection process.

## Folder Structure

The repository is organized into two main folders:

- **examples:** Jupyter Notebooks with examples demonstrating the usage of forward and backward feature selection.
- **scripts:** Contains the Python scripts implementing the forward and backward feature selection.
