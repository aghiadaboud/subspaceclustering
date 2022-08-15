import csv
import numpy as np

data = np.random.rand(100, 4)

with open('eggs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for row in data:
        writer.writerow(row) 
