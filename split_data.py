import csv
from collections import Counter

names =[]
with open('validated.tsv','r', encoding="utf8") as tsvin:
    for line in csv.reader(tsvin, delimiter='\t'):
        if line[0] not in names:
            names.append(line[0])


counts =[]
with open('validated.tsv','r', encoding="utf8") as tsvin:
    for line in csv.reader(tsvin, delimiter='\t'):
        if line[0] in names:
            counts.append(line[0])

counter = Counter(counts)



filtered = Counter(el for el in counter.elements() if counter[el] >= 10)

finalArrayIds =[]
finalArrayFiles =[]
filteredArrayIds =[]
with open('validated.tsv','r', encoding="utf8") as tsvin:
    for line in csv.reader(tsvin, delimiter='\t'):
        if line[0] in list(filtered.elements()):
            finalArrayIds.append(line[0])
            counter = Counter(finalArrayIds)
            if (line[0] in Counter(el for el in counter.elements() if counter[el] <= 10)):
                finalArrayFiles.append(line[1])
                filteredArrayIds.append(line[0])

print(len(finalArrayIds))
print(len(finalArrayFiles))
print(len(filteredArrayIds))
print(Counter(filteredArrayIds))

# with open('filtered_files.csv', "w", newline='') as filtered_files:
#     writer = csv.writer(filtered_files, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for id in filteredArrayIds:
#         writer.writerow([id])

with open('filtered_files.csv', "w", newline='') as filtered_files:
    writer = csv.writer(filtered_files, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for (id, file) in zip(filteredArrayIds,finalArrayFiles):
        writer.writerow([id,file])