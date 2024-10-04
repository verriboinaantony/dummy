# Badminton Player Image Classification

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
  - [Background Subtraction](#background-subtraction)
  - [Player Detection](#player-detection)
  - [Feature Extraction](#feature-extraction)
  - [Clustering](#clustering)
- [Assumptions and Considerations](#assumptions-and-considerations)
- [Potential Improvements](#potential-improvements)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Overview

This project aims to classify images of four badminton players from a game into individual player classes. By leveraging computer vision techniques such as background subtraction, feature extraction, and clustering, the solution isolates and identifies each player based on distinct characteristics, primarily their outfit colors.

## Prerequisites

- **Operating System**: Unix-based (Linux/macOS) recommended for shell script execution.
- **Python**: Version 3.x
- **Libraries**:
  - OpenCV
  - NumPy
  - scikit-learn

Ensure you have Python 3 installed on your system. You can verify the installation by running:

```bash
python3 --version
