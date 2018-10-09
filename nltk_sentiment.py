import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def move_to_file(data):
    with open('nltk_results.json', 'w') as outfile:
        json.dump(data, outfile,indent=4)

def sentiment_stats(master, data):
    averages = []
    for l in master:
        sum = 0
        count = 0
        for item in l:
            sum = sum + item
            count += 1
        average = float(sum)/float(count)
        averages.append("%.4f" % average)
    avgObj = {}
    avgObj['negative_average'] = averages[0]
    avgObj['neutral_average'] = averages[1]
    avgObj['positive_average'] = averages[2]
    avgObj['compound_average'] = averages[3]
    data.append(avgObj)
    move_to_file(data)

def analyze(data):
    analyzer = SentimentIntensityAnalyzer()
    compounds = []
    positives = []
    negatives = []
    neutrals = []
    for jsonObj in data:
        analysis = analyzer.polarity_scores(jsonObj['text'])
        jsonObj['negative'] = analysis['neg']
        jsonObj['neutral'] = analysis['neu']
        jsonObj['positive'] = analysis['pos']
        jsonObj['compound'] = analysis['compound']
        #used later for stats
        compounds.append(analysis['compound'])
        positives.append(analysis['pos'])
        negatives.append(analysis['neg'])
        neutrals.append(analysis['neu'])
    master = []
    master.append(negatives)
    master.append(neutrals)
    master.append(positives)
    master.append(compounds)
    sentiment_stats(master,data)
    
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
    return data
    

if __name__ == '__main__':  
    #open file
    with open('thousand.json', 'r') as f:
        loaded_data = json.load(f) 
    #preprocess data
    preprocessed_data = preprocess(loaded_data)   
    #analyze
    analyze(preprocessed_data)