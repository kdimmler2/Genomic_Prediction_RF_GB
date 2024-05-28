import os
import shutil

directory_path = 'results/iteration1'
iteration = 1

while os.path.exists(directory_path):
    iteration += 1
    directory_path = 'results/iteration' + str(iteration)

current_path = directory_path
print(current_path)

previous_iteration = iteration - 1
previous_iteration_path = 'results/iteration' + str(previous_iteration)
print(previous_iteration_path)

## Check if the destination directory exists
#if os.path.exists(destination_path):
#    # Remove the destination directory
#    os.rmdir(destination_path)

if os.path.exists('results/iteration1.bak'):
    os.rename('results/previous_iteration', current_path)
    os.rename('results/current_iteration', 'results/previous_iteration')

if not os.path.exists('results/iteration1.bak'):
    os.rename('results/iteration1', 'results/iteration1.bak')
    os.rename('results/previous_iteration', 'results/iteration1')
    os.rename('results/current_iteration', 'results/previous_iteration')

