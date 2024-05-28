from pathlib import Path

# Path to the root directory containing the iterations
root_directory = Path(".")

# Path to the output file
output_file = Path("results/eval.txt")

# List to store the mean values for each iteration
mean_values = []

# Open the output file for writing
with open(output_file, "w") as outfile:
    # Iterate over each iteration
    for i in range(1, 41):  # assuming iterations go from 1 to 40
        iteration_directory = root_directory / f"results/iteration{i}"
        file_path = iteration_directory / "evaluation_metrics.tsv"
        if file_path.is_file():
            # Open the file and read the lines
            with open(file_path, "r") as file:
                lines = file.readlines()
                # List to store the values from index 3 for each line
                values = []
                for line in lines[1:]:  # Skip the header line
                    # Split the line by tab and get the value at index 3
                    split = line.split('\t')
                    value = float(split[3])
                    values.append(value)
                # Calculate the mean of the values
                mean_value = sum(values) / len(values)
                mean_values.append(mean_value)
                # Write the output to the outfile
                outfile.write(f"Iteration{i}\t{mean_value}\n")
        else:
            print(f"File not found for iteration{i}")

# Print the mean values for each iteration
for i, mean_value in enumerate(mean_values, 1):
    print(f"Iteration {i}: Mean value = {mean_value}")

