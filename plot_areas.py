import csv
import os

from matplotlib import pyplot as plt


def main():
    file_paths = os.listdir('areas')
    areas = [[] for _ in range(9)]
    positions = []
    for idx, csv_path in enumerate(sorted(file_paths)):
        print(os.path.join('areas/', csv_path))
        with open(os.path.join('areas/', csv_path)) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i1 = -1
            for row in reader:
                if not len(row):
                    continue
                i1 += 1

                if idx == 0:
                    positions.append([float(row[1]), float(row[2])])
                    areas[i1].append(int(row[0]))
                else:
                    i2_m = 0
                    dist_m = (float(row[1]) - positions[0][0]) ** 2 + (float(row[2]) - positions[0][1]) ** 2
                    for i2 in range(1, 9):
                        dist = (float(row[1]) - positions[i2][0]) ** 2 + (float(row[2]) - positions[i2][1]) ** 2
                        if dist < dist_m:
                            i2_m = i2
                            dist_m = dist
                    areas[i2_m].append(int(row[0]))
                    print(f'{idx}: {i1} --> {i2_m}')

    print(positions)
    for i in range(9):
        plt.figure(i)
        plt.plot(range(len(areas[i])), areas[i], '-o')
        plt.title(f'Mango {i}')
        plt.xlabel('Time')
        plt.ylabel('Area')
        plt.savefig(f'graphs/{i:03d}.png')
        plt.show()
    # print(areas[0])


if __name__ == '__main__':
    main()
