from pathlib import Path

# Path to the root directory containing the iterations
root_directory = Path(".")

outfile = open('results/STDB_val_AUC_T.35.txt', 'wt')

print('iteration' + '\t' + 'AUC', file=outfile)

# Iterate over each iteration
for i in range(1, 47):  # assuming iterations go from 1 to 40
    iteration_directory = root_directory / f"results/iteration{i}"
    file_path = iteration_directory / "f1_score.txt"
    if file_path.is_file():
        infile = open(file_path)
        for line in infile:
            line = line.rstrip()
            if 'AUC:' in line:
                split = line.split(' ')
                print(str(i) + '\t' + split[1], file=outfile)
