import json

final_load = []

for i in range(0,190,10):
    runner = str(i) + str(i+10) + "google-results.json"
    with open(runner, 'r') as f:
        loaded_data = json.load(f)
    for obj in loaded_data:
        final_load.append(obj)
    

with open('converged.json', 'w') as outfile:
        json.dump(final_load, outfile,indent=4)
