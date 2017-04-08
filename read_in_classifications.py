import json

def read_in_classifications_for_training(filename):
    with open(filename) as file:
        data = file.read()

    data_values = data.split()

    classifications = []

    for i in range(len(data_values) - 1):
        classifications.append(data_values[i + 1][0])

    print(len(classifications))

    f = open('classifications_' + filename, 'w')
    json.dump(classifications, f)
    f.close()

if __name__ == "__main__":
    #read_in_classifications_for_training('train_unbalance.txt')
    read_in_classifications_for_training('val_unbalance.txt')