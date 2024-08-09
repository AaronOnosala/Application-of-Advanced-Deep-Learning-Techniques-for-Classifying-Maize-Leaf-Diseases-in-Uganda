# Import necessary libraries
import os  # Provides a way to interact with the operating system
import pandas as pd  # Used for data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations

# Define the path to the results CSV file
results_path = './runs/classify/train63/results.csv'

# Read the CSV file into a DataFrame
results = pd.read_csv(results_path)

# Create a plot for training and validation loss
plt.figure()  # Create a new figure for plotting
plt.plot(results['                  epoch'], results['             train/loss'], label='train loss')  # Plot training loss
plt.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')  # Plot validation loss
plt.grid()  # Add grid lines for better readability
plt.title('Loss vs epochs')  # Set the title of the plot
plt.ylabel('loss')  # Label for the y-axis
plt.xlabel('epochs')  # Label for the x-axis
plt.legend()  # Add a legend to differentiate between training and validation loss

# Create a plot for validation accuracy
plt.figure()  # Create a new figure for plotting
plt.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)  # Plot validation accuracy as a percentage
plt.grid()  # Add grid lines for better readability
plt.title('Validation accuracy vs epochs')  # Set the title of the plot
plt.ylabel('accuracy (%)')  # Label for the y-axis
plt.xlabel('epochs')  # Label for the x-axis

# Display all plots
plt.show()
