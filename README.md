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

Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/badminton-player-classification.git
cd badminton-player-classification
Install Required Python Libraries

It's recommended to use a virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Then install the dependencies:

bash
Copy code
pip install -r requirements.txt
Alternatively, install directly:

bash
Copy code
pip install opencv-python numpy scikit-learn
Prepare the Directory Structure

Ensure your project directory contains the following:

scss
Copy code
badminton-player-classification/
├── execute.sh
├── classify_players.py
├── top_two_players/ (folder)
├── bot_two_players/ (folder)
└── court_image.jpg
top_two_players/: Folder containing images of the top two players.
bot_two_players/: Folder containing images of the bottom two players.
court_image.jpg: Reference image of the badminton court for background subtraction.
Directory Structure
lua
Copy code
badminton-player-classification/
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
output/: This directory will be created by the execute.sh script and will contain subdirectories for each classified player.
Usage
Make the Shell Script Executable

bash
Copy code
chmod +x execute.sh
Run the Shell Script

bash
Copy code
./execute.sh
This script performs the following actions:

Creates the necessary output directories:
output/Player1/
output/Player2/
output/Player3/
output/Player4/
Executes the Python script classify_players.py to process and classify the images.
Check the Output

After successful execution, the output/ directory will contain four subdirectories, each corresponding to a player, populated with the classified images.

Implementation Details
Background Subtraction
Function: subtract_background(image, background)
Process:
Converts both the game image and the court image to grayscale.
Computes the absolute difference to highlight moving objects (players).
Applies thresholding to obtain a binary image that isolates players from the background.
Player Detection
Function: extract_player_regions(thresh_image)
Process:
Detects contours in the thresholded image.
Filters out small contours based on a predefined area threshold to eliminate noise.
Extracts bounding rectangles around the detected player regions.
Feature Extraction
Function: compute_color_histogram(image)
Process:
Converts the cropped player image to HSV color space for better color segmentation.
Computes a 3D color histogram with bins [8, 8, 8] for the HSV channels.
Normalizes and flattens the histogram to create a feature vector representing the player's color distribution.
Clustering
Function: process_images(folder_path, court_image, player_indices)
Process:
Processes each image in the specified folder (top_two_players or bot_two_players).
Extracts features from detected player regions.
Applies K-Means clustering with k=2 to group images into two clusters per folder.
Assigns images to respective player folders based on cluster labels.
Assumptions and Considerations
Distinct Player Outfits: The method assumes that each player wears distinct colors, making color histograms effective for differentiation.
Consistent Lighting Conditions: Uniform lighting across images ensures reliable background subtraction.
Area Threshold: The area threshold in player detection (500 pixels) may need adjustment based on image resolution and player sizes.
Execution Time: The solution is optimized to run within a 2-minute execution time constraint.
Potential Improvements
Advanced Feature Extraction: Incorporate more sophisticated features like Histogram of Oriented Gradients (HOG) or employ deep learning models for better differentiation, especially if players have similar colors.
Enhanced Background Subtraction: Utilize advanced background subtraction methods such as MOG2 or KNN algorithms provided by OpenCV for more robust player isolation.
Robust Error Handling: Implement more comprehensive error handling to manage cases where no players are detected or images are corrupted.
Parameter Tuning: Adjust clustering parameters or explore other clustering algorithms (e.g., DBSCAN) if K-Means does not yield optimal results.
Real-time Processing: Adapt the solution for real-time video processing if needed.
Troubleshooting
Court Image Not Found: Ensure that court_image.jpg is correctly placed in the project directory and the filename matches.
Insufficient Player Detection: If not enough players are detected, consider adjusting the area threshold in extract_player_regions or improving image quality.
Library Installation Issues: Verify that all required libraries are installed. Use pip list to check installed packages.
Script Permissions: Ensure that execute.sh has execute permissions (chmod +x execute.sh).
License
This project is licensed under the MIT License.

Contact
For any questions or support, please contact:

Name: Your Name
Email: your.email@example.com
GitHub: yourusername
Happy Coding!

yaml
Copy code

---

### 2. `execute.sh`

```bash
#!/bin/bash

# Create output directories
mkdir -p output/Player1 output/Player2 output/Player3 output/Player4

# Run the Python script
python3 classify_players.py
Make sure to make this script executable:

bash
Copy code
chmod +x execute.sh
3. classify_players.py
python
Copy code
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import shutil

def subtract_background(image, background):
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Subtract background
    diff = cv2.absdiff(gray_image, gray_background)

    # Thresholding to get binary image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return thresh

def extract_player_regions(thresh_image):
    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume players are the largest contours
    player_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Adjust area threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            player_regions.append((x, y, w, h))

    return player_regions

def compute_color_histogram(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()

def process_images(folder_path, court_image, player_indices):
    features = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Background subtraction
            thresh = subtract_background(image, court_image)

            # Player detection
            player_regions = extract_player_regions(thresh)

            for idx, (x, y, w, h) in enumerate(player_regions):
                # Crop player region
                player_image = image[y:y+h, x:x+w]

                # Feature extraction
                hist = compute_color_histogram(player_image)

                features.append(hist)
                image_paths.append((image_path, idx, player_image))

    # Clustering
    if len(features) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(features)

        # Organize images into player folders
        for (img_path, idx, player_img), label in zip(image_paths, labels):
            player_folder = f'output/Player{player_indices[label]}'
            filename = os.path.basename(img_path)
            save_path = os.path.join(player_folder, filename)

            # Save the cropped player image
            cv2.imwrite(save_path, player_img)
    else:
        print(f"Not enough player images found in {folder_path}.")

def main():
    # Load the court image
    court_image_path = 'court_image.jpg'  # Replace with your court image filename
    court_image = cv2.imread(court_image_path)

    if court_image is None:
        print("Court image not found.")
        return

    # Process top two players
    process_images('top_two_players', court_image, player_indices=[1, 2])

    # Process bottom two players
    process_images('bot_two_players', court_image, player_indices=[3, 4])

if __name__ == "__main__":
    main()
4. requirements.txt
It's good practice to include a requirements.txt file for easy installation of dependencies.

text
Copy code
opencv-python
numpy
scikit-learn
5. court_image.jpg
Ensure you have a reference image of the badminton court named court_image.jpg placed in your project directory. This image is used for background subtraction to isolate the players from the court.

6. Directory Setup
Create the following directories and add your player images accordingly:

Top Two Players: top_two_players/
Add images: player1_image1.jpg, player1_image2.jpg, etc.
Bottom Two Players: bot_two_players/
Add images: player3_image1.jpg, player3_image2.jpg, etc.
Your final project structure should look like this:

lua
Copy code
badminton-player-classification/
├── execute.sh
├── classify_players.py
├── requirements.txt
├── top_two_players/
│   ├── player1_image1.jpg
│   ├── player1_image2.jpg
│   └── ...
├── bot_two_players/
│   ├── player3_image1.jpg
│   ├── player3_image2.jpg
│   └── ...
├── court_image.jpg
├── output/ (will be created after running execute.sh)
└── README.md
7. Running the Project
Install Dependencies

If you haven't already, install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Make the Shell Script Executable

bash
Copy code
chmod +x execute.sh
Execute the Script

bash
Copy code
./execute.sh
This will create the output/ directory with four subdirectories (Player1, Player2, Player3, Player4) containing the classified player images.

8. Additional Notes
Adjusting Parameters: You may need to tweak the area threshold in the extract_player_regions function based on your image resolutions and the size of the players in the images.
Error Handling: Ensure that all images are correctly named and placed in their respective directories to avoid processing errors.
Extending Functionality: For improved accuracy, consider integrating more advanced feature extraction techniques or machine learning models.
