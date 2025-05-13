import pickle


with open("tmp_results.pkl", "rb") as f:
    results = pickle.load(f)

print(results[0])
