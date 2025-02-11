Music Recommendation System

Overview

This project is a machine learning-based music recommendation system that predicts songs based on user inputs such as:

Time of day (Morning, Afternoon, Evening, Night)

Steps (Number of steps taken by the user)

Temperature (Current temperature in degrees Celsius)

Wind speed (Wind speed in km/h)

Genre/Mood (Chill, Upbeat, etc.)

The model is trained using a dataset containing 5000 music entries, and it uses a Neural Network built with PyTorch to predict a song that matches the given inputs.

Features

Uses a Neural Network (NN) for classification.

Case-insensitive user input handling.

Handles unknown inputs with error messages and valid options.

Supports dynamic user input validation.

Requirements

Ensure you have the following dependencies installed:

pip install torch torchvision pandas numpy scikit-learn matplotlib tqdm accelerate

How It Works

The model is trained on a dataset where songs are mapped to input conditions.

A user enters the required parameters, and the model predicts the most suitable song.

The output is a recommended song title based on the given inputs.

Running the Project

To run the program, execute the following command:

python model.py

User Input Format

The program expects the user to enter values in the following format:

Enter time of day (Morning, Afternoon, Evening, Night): afternoon

Enter number of steps: 768

Enter temperature: 8

Enter wind speed: 17

Enter genre/mood (Chill, Upbeat, etc.): hip-hop

Example Output

Recommended music: Zaroori Tha - Rahat Fateh Ali Khan

Model Details

The neural network consists of:

Input Layer: 5 features

Two hidden layers with 512 neurons each

ReLU activation functions

Output Layer: 15 possible song classes

Training Configuration

Loss Function: CrossEntropyLoss (multi-class classification)

Optimizer: Adam (learning rate = 0.0001)

Epochs: 100

Batch Size: 10

Data Split: 80% training, 20% testing





License

This project is open-source and free to use for learning and development purposes.

Author

Developed by Malik Haroon Khokhar.

