import pickle


distribution_file_1 = input("Disrt 1:")
distribution_file_2 = input("Distr 2:")

with open(distribution_file_1, "rb") as f:
    distr1 = pickle.load(f)

with open(distribution_file_2, "rb") as f:
    distr2 = pickle.load(f)

print(distr1)
print(distr2)