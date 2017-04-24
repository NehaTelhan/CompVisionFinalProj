import json
import numpy

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

def retrieve_classifications(filename, imgname):
    f = open(filename, 'r')
    classification_list = f.read().replace("[", "")
    classification_list = classification_list.replace("]", "")
    classification_list = classification_list.split(",")
    imgname = imgname.split("_")
    imgnum = imgname[1].split(".")
    if filename == "val_unbalance.txt":
        imgnum = int(imgnum[0]) - 680674
        classification = classification_list[imgnum]
    else:
        classification = classification_list[int(imgnum[0])]
    classification = classification.replace("\"", "")
    num = int(classification)
    if num == 0:
        num = 0
    else:
        num = 1
    return num

def convert_zero_classifications_to_one(classifications):
    fixed_classifications = numpy.zeros(len(classifications))
    for i in classifications:
        if i == 0:
            fixed_classifications[i] = 1
        else:
            fixed_classifications[i] = -1
    return fixed_classifications

if __name__ == "__main__":
    #read_in_classifications_for_training('train_unbalance.txt')
    #read_in_classifications_for_training('val_unbalance.txt')
    # print(retrieve_classifications("classifications_train_unbalance.txt", "text_0.jpg"))
    pass