import csv

adj = []
star = []

adj_0_1 = []
star_0_1 = []

star_agg = 0
adj_agg = 0

with open('Adjusted_ModelBasedCF.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in readCSV:
        temp = {}
        temp["user_id"] = row[0]
        temp["business_id"] = row[1]
        temp["star"] = row[2]
        adj.append(temp)
        adj_agg += abs(float(row[2]))
        if abs(float(row[2])) < 1:
            adj_0_1.append(count)
        count += 1

with open('Star_ModelBasedCF.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in readCSV:
        temp = {}
        temp["user_id"] = row[0]
        temp["business_id"] = row[1]
        temp["star"] = row[2]
        star.append(temp)
        star_agg += abs(float(row[2]))
        if abs(float(row[2])) < 1:
            star_0_1.append(count)
        count += 1
        
intersect = [value for value in adj_0_1 if value in star_0_1] 
        

print("ADJ LENGTH")
print(len(adj))
print("STAR LENGTH")
print(len(star))
print("ADJ_0_1 LENGTH")
print(len(adj_0_1))
print("STAR_0_1 LENGTH")
print(len(star_0_1))
print("INTERSECTION")
print(len(intersect))
print("STAR AGG")
print(star_agg)
print("ADJ AGG")
print(adj_agg)


