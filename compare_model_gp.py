import os
import pickle
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--results-dir', required=True)

args = parser.parse_args()

model_dirs = os.listdir(args.results_dir)
model_results = []

for model_dir in model_dirs:
    dir = os.path.join(args.results_dir, model_dir)
    results_file = os.path.join(dir, 'results.pkl')
    if os.path.isfile(results_file):
        results = pickle.load(open(results_file, 'rb'))
        model_results.append(results)
    else:
        print(f'Skipping {results_file}, file not found')

plt.figure()

for results in model_results:
    plt.plot(results['epochs'], results['validation_losses'], label=f'{results["args"].model}')

plt.xlabel('epochs')
plt.ylabel('validation predictive NLL')

plt.legend()
plt.show()