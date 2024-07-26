import os
import re
import numpy as np
def pad_array_to_length(array, desired_length):
    """
    Pads the array to the desired length by repeating the last value.
    
    Parameters:
    - array: The input numpy array to pad.
    - desired_length: The desired length of the array after padding.
    
    Returns:
    - A numpy array padded to the desired length.
    """
    current_length = np.shape(array)[0]
    if current_length < desired_length:
        padding_length = desired_length - current_length
        last_value = array[-1]
        padding_values = np.full(padding_length, last_value)
        array = np.append(array, padding_values)
    return array


# Directory containing the input files
input_directory = 'prints'

# Output file paths
normal_mse_file_path0 = '../notebooks/resultados/Lines_normal_MSE_f1.txt'
closest_mse_file_path0 = '../notebooks/resultados/Lines_closest_MSE_f1.txt'
mean_mse_file_path0 = '../notebooks/resultados/Lines_mean_MSE_f1.txt'

# Output file paths
normal_mse_file_path1 = '../notebooks/resultados/Lines_normal_MSE_f2.txt'
closest_mse_file_path1 = '../notebooks/resultados/Lines_closest_MSE_f2.txt'
mean_mse_file_path1 = '../notebooks/resultados/Lines_mean_MSE_f2.txt'

# Regular expressions to match the lines
normal_mse_pattern = re.compile(r'normal_MSE\s+([\d.]+)')
closest_mse_pattern = re.compile(r'closest_MSE\s+([\d.]+)')

# Initialize lists to hold the extracted MSE values for f0
normal_mse_values0 = []
closest_mse_values0 = []

# Initialize lists to hold the extracted MSE values for f1
normal_mse_values1 = []
closest_mse_values1 = []

a=[]
# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.startswith('Lines') and filename.endswith('1.dat'):
        file_path = os.path.join(input_directory, filename)
        # Read the input file and extract the MSE values
        with open(file_path, 'r') as file:
            normal_mse_file_values = []
            closest_mse_file_values = []
            for line in file:
                normal_mse_match = normal_mse_pattern.search(line)
                closest_mse_match = closest_mse_pattern.search(line)
                if normal_mse_match:
                    normal_mse_file_values.append(float(normal_mse_match.group(1)))
                if closest_mse_match:
                    closest_mse_file_values.append(float(closest_mse_match.group(1)))

            # Correct if problems
            desired_length = 8200 #para este caso en concreto (200 pop size 41 generations)
            current_length = np.shape(normal_mse_file_values)[0]
            if current_length < desired_length:
                normal_mse_file_values = pad_array_to_length(normal_mse_file_values,desired_length)
            current_length = np.shape(closest_mse_file_values)[0]
            if current_length < desired_length:
                closest_mse_file_values = pad_array_to_length(closest_mse_file_values,desired_length)

            normal_mse_values0.append(normal_mse_file_values)
            closest_mse_values0.append(closest_mse_file_values)
    
    if filename.startswith('Lines') and filename.endswith('2.dat'):
        file_path = os.path.join(input_directory, filename)
        # Read the input file and extract the MSE values
        with open(file_path, 'r') as file:
            normal_mse_file_values = []
            closest_mse_file_values = []
            for line in file:
                normal_mse_match = normal_mse_pattern.search(line)
                closest_mse_match = closest_mse_pattern.search(line)
                if normal_mse_match:
                    normal_mse_file_values.append(float(normal_mse_match.group(1)))
                if closest_mse_match:
                    closest_mse_file_values.append(float(closest_mse_match.group(1)))
                    
            # Correct if problems
            desired_length = 8200 #para este caso en concreto (200 pop size 41 generations)
            current_length = np.shape(normal_mse_file_values)[0]
            if current_length < desired_length:
                normal_mse_file_values = pad_array_to_length(normal_mse_file_values,desired_length)
            current_length = np.shape(closest_mse_file_values)[0]
            if current_length < desired_length:
                closest_mse_file_values = pad_array_to_length(closest_mse_file_values,desired_length)
          
            normal_mse_values1.append(normal_mse_file_values)
            closest_mse_values1.append(closest_mse_file_values)


# Calculate the mean for each line

normal_mse_means0 = np.mean(normal_mse_values0, axis=0)
closest_mse_means0 = np.mean(closest_mse_values0, axis=0)

# Write the extracted values to the output files
with open(normal_mse_file_path0, 'w') as file:
    for values in normal_mse_values0:
        file.write(' '.join(map(str, values)) + '\n')

with open(closest_mse_file_path0, 'w') as file:
    for values in closest_mse_values0:
        file.write(' '.join(map(str, values)) + '\n')

# Write the mean values to the mean MSE file
with open(mean_mse_file_path0, 'w') as file:
    for normal_mean, closest_mean in zip(normal_mse_means0, closest_mse_means0):
        file.write(f'{normal_mean} {closest_mean}\n')






normal_mse_means1 = np.mean(normal_mse_values1, axis=0)
closest_mse_means1 = np.mean(closest_mse_values1, axis=0)

# Write the extracted values to the output files
with open(normal_mse_file_path1, 'w') as file:
    for values in normal_mse_values1:
        file.write(' '.join(map(str, values)) + '\n')

with open(closest_mse_file_path1, 'w') as file:
    for values in closest_mse_values1:
        file.write(' '.join(map(str, values)) + '\n')

# Write the mean values to the mean MSE file
with open(mean_mse_file_path1, 'w') as file:
    for normal_mean, closest_mean in zip(normal_mse_means1, closest_mse_means1):
        file.write(f'{normal_mean} {closest_mean}\n')

print("Extraction and mean calculation complete. The normal MSE values are saved in 'normal_MSE.txt', the closest MSE values are saved in 'closest_MSE.txt', and the mean MSE values are saved in 'mean_MSE.txt'.")
