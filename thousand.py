import ijson
import json
import io

filename = 'yelp_academic_dataset_review.json'
outputfile = 'thousand.json'
jsonObjects = []

with open(filename, encoding="UTF-8") as json_file:
    cursor = 0
    count = 0
    for line_number, line in enumerate(json_file):
        jsonObj = {}
        line_as_file = io.StringIO(line)
        
        # Use a new parser for each line
        json_parser = ijson.parse(line_as_file)
        for prefix, type, value in json_parser:
            jsonObj[prefix] = value
        cursor += len(line)
        count += 1
        
        if '' in jsonObj:
            del jsonObj['']
        jsonObjects.append(jsonObj)
        if count == 1000:
            break

with open(outputfile, 'w') as outfile:
    json.dump(jsonObjects, outfile,indent=4)


