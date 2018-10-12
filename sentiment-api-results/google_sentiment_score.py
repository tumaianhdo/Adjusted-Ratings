#do google auth login first in command line

import json
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def move_to_file(data):
    with open('google_results.json', 'w') as outfile:
        json.dump(data, outfile,indent=4)

def analyze(data):
    
    client = language.LanguageServiceClient()
    
    for jsonObj in data:  
        document = types.Document(
            content=jsonObj['text'],
            type=enums.Document.Type.PLAIN_TEXT)
        
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        jsonObj['sentiment-google'] = sentiment.score
        jsonObj['magnitude-google'] = sentiment.magnitude

    # Print the results
    print(data[0])
    move_to_file(data)

def preprocess(data):
    for jsonObj in data:
        if 'date' in jsonObj:
            del jsonObj['date']
        if 'useful' in jsonObj:
            del jsonObj['useful']
        if 'funny' in jsonObj:
            del jsonObj['funny']
        if 'cool' in jsonObj:
            del jsonObj['cool']
        #adding temp values
        jsonObj['sentiment-google'] = 0
        jsonObj['magnitude-google'] = 0
    return data


if __name__ == '__main__':
    print(0)
    #open file
    with open('thousand.json', 'r') as f:
        loaded_data = json.load(f)
    print(1)
    #preprocess data
    preprocessed_data = preprocess(loaded_data)
    print(2)
    
    #analyze
    analyze(preprocessed_data)