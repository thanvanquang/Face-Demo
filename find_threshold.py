import json
import numpy as np
import itertools
import matplotlib.pyplot as plt

def euclid_distance(emb1, emb2):
    return np.linalg.norm(emb1-emb2, keepdims=False)

data_file = './DB/VTS_backbone.txt'
f = open(data_file, 'r')
data_set = json.loads(f.read())

for i, person in enumerate(data_set.keys()):
    person_data = data_set[person]
    person_data = np.array(person_data)
    number_vector = person_data.shape[0]
    person_names = [person] * number_vector
    if i == 0:
        dataset = person_data
        people_names = person_names
    else:
        dataset = np.concatenate((dataset, person_data), axis=0)
        people_names = list(itertools.chain(people_names, person_names))

similar = []
difference = []
for j in range(len(people_names)):
    for k in range(len(people_names)):
        distance = euclid_distance(dataset[j], dataset[k])
        if people_names[j] == people_names[k]:
            similar.append(distance)
        else:
            difference.append(distance)


y_similar = np.array(similar)
y_difference = np.array(difference)

# y_similar = np.sort(y_similar)
# y_difference = np.sort(y_difference)

name_plot = data_file.split("/")[-1].split(".")[0]

plt.plot(y_similar)
# plt.plot(y_difference)
plt.title(name_plot)
plt.show()





