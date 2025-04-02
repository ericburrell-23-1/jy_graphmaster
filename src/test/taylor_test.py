import pickle

# Open the file in binary read mode
with open("time.pkl", "rb") as f:
    my_loaded_variable = pickle.load(f)

# Print the loaded data
print(my_loaded_variable)