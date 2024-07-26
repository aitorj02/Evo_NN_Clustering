import os
import re
import numpy as np
import matplotlib.pyplot as plt


def extract_values(file_path, keyword):
    """Extract values from a file based on the given keyword."""
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            if keyword in line:
                # Extract the first float number in the line after the keyword
                match = re.search(rf'{keyword}\s+(\d+\.?\d*)', line)
                if match:
                    values.append(float(match.group(1)))
    return values

def calculate_mean_values(directory, keyword, file_pattern):
    """Calculate mean values for each position across all files in the directory based on the given keyword."""
    all_values = []
    file_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(file_pattern):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                values = extract_values(file_path, keyword)
                all_values.append(values)
                file_count += 1
    
    if not all_values:
        raise ValueError(f"No values found for keyword '{keyword}' in the files.")
    
    # Transpose to get the values for each position across all files
    all_values = np.array(all_values).T
    mean_values = np.mean(all_values, axis=1)
    
    return mean_values


def cumulative_minimum(values):
    """Calculate cumulative minimum values."""
    cum_min_values = np.minimum.accumulate(values)
    return cum_min_values

# Directory containing the files
directory = 'prints'
file_pattern = ''  # File pattern to filter files

# Calculate the mean MSE values
mean_mse_values = calculate_mean_values(directory, 'normal_MSE', file_pattern)

# Calculate the mean 'closest' values
mean_closest_values = calculate_mean_values(directory, 'closest_MSE', file_pattern)

# Calculate cumulative minimum values
cum_min_mse_values = cumulative_minimum(mean_mse_values)
cum_min_closest_values = cumulative_minimum(mean_closest_values)

# # Print the mean values for comparison
# for i, (mean_mse, mean_closest) in enumerate(zip(mean_mse_values, mean_closest_values)):
#     print(f"Position {i+1}: Mean MSE = {mean_mse}, Mean closest = {mean_closest}")
print(min(cum_min_mse_values))

# Plotting the results
positions = range(1, len(mean_mse_values) + 1)
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.plot(positions, cum_min_mse_values, marker='o', linestyle='-', color='b', label='Cumulative Min MSE')
plt.plot(positions, cum_min_closest_values, marker='o', linestyle='-', color='r', label='Cumulative Min closest')

plt.xlabel('Position', fontsize=14)
plt.ylabel('Cumulative Minimum Value', fontsize=14)
plt.title('Comparison of Cumulative Min MSE and Cumulative Min closest Values')
plt.legend()
plt.grid(True)
plt.show()