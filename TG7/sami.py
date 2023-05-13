import numpy as np


data=np.load("attack_data.npy",allow_pickle=True)
print(data.shape)
last_column = data[:, -1]  # Extract the last column
last_column[last_column == 'Normal'] = 0
last_column[last_column == 'Attack'] = 1


# Update the modified column back into the original array
data[:, -1] = last_column
data_normal = data[last_column == 0]
data_attack = data[last_column == 1]


batch_size = 32


num_batches_normal = len(data_normal) // batch_size
num_batches_attack = len(data_attack) // batch_size


# Trim the arrays to have an equal number of points for each batch
data_normal = data_normal[:num_batches_normal * batch_size]
data_attack = data_attack[:num_batches_attack * batch_size]
print(data_normal[:,-1])
print(data_attack[:,-1])
# Reshape the arrays into batches


data_normal_batches = np.reshape(data_normal, (num_batches_normal, batch_size, data_normal.shape[1]))
data_attack_batches = np.reshape(data_attack, (num_batches_attack, batch_size, data_attack.shape[1]))


concatenated_data = np.concatenate((data_normal_batches, data_attack_batches), axis=0)
print(concatenated_data.shape)
# print(concatenated_data[:,-1])
np.random.shuffle(concatenated_data)
np.save('processed_attack_data.npy', concatenated_data)



