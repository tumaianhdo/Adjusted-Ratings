import json
import matplotlib.pyplot as plt

users = {}
businesses = {}
differences = []
differences_size_over_1 = []
u_differences = []
u_differences_size_over_1 = []

# =============================================================================
# Categories (5, interval[float64]): 
# [(-0.902, -0.54] < (-0.54, -0.18] < (-0.18, 0.18] 
#     < (0.18, 0.54] < (0.54, 0.9]]
# =============================================================================

def bucketize(score):
    if score <= -0.54:
        return 1
    elif score <= -0.18:
        return 2
    elif score <= 0.18:
        return 3
    elif score <= 0.54:
        return 4
    else: 
        return 5

with open('google_results.json', 'r') as infile:
    objects = json.loads(infile.read())
    for obj in objects:
        if obj["user_id"] in users:
            users[obj["user_id"]]["stars"].append(obj["stars"]) 
            users[obj["user_id"]]["sentiment"].append(bucketize(obj["sentiment-google"])) 
        else:
            users[obj["user_id"]] = {}
            users[obj["user_id"]]["stars"] = [obj["stars"]]
            users[obj["user_id"]]["sentiment"] = [bucketize(obj["sentiment-google"])]
        if obj["business_id"] in businesses:
            businesses[obj["business_id"]]["stars"].append(obj["stars"]) 
            businesses[obj["business_id"]]["sentiment"].append(bucketize(obj["sentiment-google"])) 
        else:
            businesses[obj["business_id"]] = {}
            businesses[obj["business_id"]]["stars"] = [obj["stars"]]
            businesses[obj["business_id"]]["sentiment"] = [bucketize(obj["sentiment-google"])]

# all listed now
for user in users:
    users[user]["numReviews"] = len(users[user]["stars"])
    difference = 0
    for i in range(len(users[user]["stars"])):
        difference = difference + abs(users[user]["stars"][i] - users[user]["sentiment"][i])
    users[user]["difference"] = difference/len(users[user]["stars"])
    u_differences.append(users[user]["difference"])
    if(users[user]["numReviews"] > 1):
        u_differences_size_over_1.append(users[user]["difference"])
for business in businesses:
    businesses[business]["numReviews"] = len(businesses[business]["stars"])
    difference = 0
    for i in range(len(businesses[business]["stars"])):
        difference = difference + abs(businesses[business]["stars"][i] - businesses[business]["sentiment"][i])
    businesses[business]["difference"] = difference/len(businesses[business]["stars"])
    differences.append(businesses[business]["difference"])
    if(businesses[business]["numReviews"] > 1):
        differences_size_over_1.append(businesses[business]["difference"])
        
print("ALL BUSINESS DIFFERENCES SHOWN")
plt.hist(differences)
plt.show()
print("SIZE > 1 BUSINESS DIFFERENCES SHOWN")
plt.hist(differences_size_over_1)
plt.show()
print("ALL USER DIFFERENCES SHOWN")
plt.hist(u_differences)
plt.show()
print("SIZE > 1 USER DIFFERENCES SHOWN")
plt.hist(u_differences_size_over_1)
plt.show()


# =============================================================================
# print(users)
# print("---------------------------------")
# print(businesses)
# =============================================================================