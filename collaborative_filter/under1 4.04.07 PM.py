import csv

star = []
adj = []

with open('star10_under1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in readCSV:
        star.append(row)
   
with open('adj10_Under1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in readCSV:
        adj.append(row)
        
print("ADJ LENGTH")
print(len(adj))
print("STAR LENGTH")
print(len(star))

intersect = [value for value in adj if value in star] 

print("INTERSECTION")
print(len(intersect))