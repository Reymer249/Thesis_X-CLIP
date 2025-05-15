import pickle


with open("batch_counter.pkl", "rb") as f:
    counter = pickle.load(f)

with open("tmp_results.pkl", "rb") as res_f:
    results = pickle.load(res_f)

with open("hard_positives_distribution.pkl", "rb") as f:
    distr = pickle.load(f)

# print("Counter", counter)
# print("Len captions:", len(results))
print(distr)