# Badminton Player Image Classification

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
pip install -r requirements.txt
```
## OR
```bash
pip install opencv-python numpy scikit-learn
```
## Prepare the Directory Structure

Ensure your project directory contains the following:
```implementation_1/
├── execute.sh
├── classify_players.py
├── top_two_players/ (folder)
├── bot_two_players/ (folder)
└── court_image.jpg
```
top_two_players/: Folder containing images of the top two players.
bot_two_players/: Folder containing images of the bottom two players.
court_image.jpg: Reference image of the badminton court for background subtraction.

## Directory Structure

```
implementation_1/
├── execute.sh
├── classify_players.py
├── top_two_players/
│   ├── player1_image1.jpg
│   ├── player1_image2.jpg
│   └── ...
├── bot_two_players/
│   ├── player3_image1.jpg
│   ├── player3_image2.jpg
│   └── ...
├── court_image.jpg
├── output/
│   ├── Player1/
│   ├── Player2/
│   ├── Player3/
│   └── Player4/
└── README.md
```
output/: This directory will be created by the execute.sh script and will contain subdirectories for each classified player.

## Usage
Make the Shell Script Executable
```bash
chmod +x execute.sh
./execute.sh
```
### This script performs the following actions:

Creates the necessary output directories:
output/Player1/
output/Player2/
output/Player3/
output/Player4/
Executes the Python script classify_players.py to process and classify the images.
Check the Output

After successful execution, the output/ directory will contain four subdirectories, each corresponding to a player, populated with the classified images.

