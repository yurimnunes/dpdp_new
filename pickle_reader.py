import pickle
#load vrp_nazari100_train_seed42-lkh
with open('vrp_nazari100_train_seed42/vrp_nazari100_train_seed42-lkh.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[0])
