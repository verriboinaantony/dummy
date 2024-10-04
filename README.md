Overview
This project aims to classify images of four badminton players from a game into individual player classes. By leveraging computer vision techniques such as background subtraction, feature extraction, and clustering, the solution isolates and identifies each player based on distinct characteristics, primarily their outfit colors.

Prerequisites
Operating System: Unix-based (Linux/macOS) recommended for shell script execution.
Python: Version 3.x
Libraries:
OpenCV
NumPy
scikit-learn
Ensure you have Python 3 installed on your system. You can verify the installation by running:

bash
Copy code
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
