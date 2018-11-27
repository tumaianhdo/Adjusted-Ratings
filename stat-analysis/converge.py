import json

final_load = []

for i in range(0,190,10):
    runner = "user_normalizedSentStars_" + str(i) + str(i+10) + ".json"
    print("Opening %s" % runner)
    with open(runner, 'r') as f:
        loaded_data = json.load(f)
    for obj in loaded_data:
        final_load.append(obj)
    

with open('converged_stars.json', 'w') as outfile:
        json.dump(final_load, outfile,indent=4)
