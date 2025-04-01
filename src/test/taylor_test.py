import sys


#taylor code

import pickle
with open('serialized_input_revised.pkl', 'rb') as f:
    data = pickle.load(f)
 

filtered_dict = {k: v for k, v in data['dict_edge_constraint_coefs'].items() if np.abs(v) >.0001}
data['dict_edge_constraint_coefs']=filtered_dict

with open('filtered_input.pkl', 'wb') as f:
    pickle.dump(data, f)

my_sum=0
for key in data:
    my_size = sys.getsizeof(data[key])
    print(f"Key: {key}, Size: {my_size} bytes")
    my_sum=my_sum+my_size

for key in data:
    my_size = sys.getsizeof(data[key])
    print(f"Key: {key}, Size: {my_size/my_sum} bytes")
tmp=data.copy()