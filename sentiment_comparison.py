import json
import statistics

# =============================================================================
# STAT FUNCTIONS
# =============================================================================
def calcAvg(master):
    averages = []
    for l in master:
        sum = 0
        count = 0
        for item in l:
            sum = sum + item
            count += 1
        average = float(sum)/float(count)
        averages.append(float("%.4f" % average))
    return averages
def calcMax(master):
    maxes = []
    for l in master:
        maxes.append(float("%.4f" % max(l)))
    return maxes
def calcMin(master):
    mins = []
    for l in master:
        mins.append(float("%.4f" % min(l)))
    return mins
def calcStdev(master):
    stdevs = []
    for l in master:
        stdevs.append(float("%.4f" % statistics.stdev(l)))
    return stdevs
def calcMed(master):
    medians = []
    for l in master:
        medians.append(float("%.4f" % statistics.median(l)))
    return medians

# =============================================================================
# GENERAL SUBMASTER FUNCTION
# =============================================================================
def generalNLTK(data):
    #preprocess to list of lists
    compounds = []
    positives = []
    negatives = []
    neutrals = []
    for jsonObj in data:
        compounds.append(jsonObj['compound'])
        positives.append(jsonObj['positive'])
        negatives.append(jsonObj['negative'])
        neutrals.append(jsonObj['neutral'])
    master = []
    master.append(negatives)
    master.append(neutrals)
    master.append(positives)
    master.append(compounds)
    #get stats
    averages = calcAvg(master)
    numReviews = len(compounds)
    std = calcStdev(master)
    median = calcMed(master)
    top = calcMax(master)
    bottom = calcMin(master)
    #output
    output = {}
    output['numberReviews'] = numReviews
    output['averages'] = dictNLTK(averages)
    output['stdev'] = dictNLTK(std)
    output['median'] = dictNLTK(median)
    output['max'] = dictNLTK(top)
    output['min'] = dictNLTK(bottom)
    return output
def generalGOOG(data):
    #preprocess to list of lists
    scores = []
    magnitudes = []
    for jsonObj in data:
       scores.append(float("%.4f" % jsonObj['sentiment-google']))
       magnitudes.append(float("%.4f" % jsonObj['magnitude-google']))
    master = []
    master.append(scores)
    master.append(magnitudes)
    #get stats
    averages = calcAvg(master)
    numReviews = len(scores)
    std = calcStdev(master)
    median = calcMed(master)
    top = calcMax(master)
    bottom = calcMin(master)
    #output
    output = {}
    output['numberReviews'] = numReviews
    output['averages'] = dictGOOG(averages)
    output['stdev'] = dictGOOG(std)
    output['median'] = dictGOOG(median)
    output['max'] = dictGOOG(top)
    output['min'] = dictGOOG(bottom)
    return output
    
# =============================================================================
# GROUP BY USER SUBMASTER FUNCTION
# =============================================================================
def userGroupNLTK(data):
    #preprocess into object with users as keys
    #retains all information
    master = {}
    for jsonObj in data:
        if jsonObj['user_id'] in master:
            prefix = master[jsonObj['user_id']]['data']
            prefix[0].append(jsonObj['negative'])
            prefix[1].append(jsonObj['neutral'])
            prefix[2].append(jsonObj['positive'])
            prefix[3].append(jsonObj['compound'])
        else:
            master[jsonObj['user_id']] = {'data':[
                [jsonObj['negative']],
                [jsonObj['neutral']],
                [jsonObj['positive']],
                [jsonObj['compound']]
            ]
    }
    #add stats to each object
    for key in master:
        prefix = master[key]
        prefix['numberReviews'] = len(prefix['data'][0])
        prefix['averages'] = dictNLTK(calcAvg(prefix['data']))
        if len(prefix['data'][0]) > 1: prefix['stdev'] = dictNLTK(calcStdev(prefix['data']))
        else: prefix['stdev'] = {'negative': 0, 'neutral': 0, 'positive': 0, 'compound': 0}
        prefix['median'] = dictNLTK(calcMed(prefix['data']))
        prefix['max'] = dictNLTK(calcMax(prefix['data']))
        prefix['min'] = dictNLTK(calcMin(prefix['data']))
    return master
def userGroupGOOG(data):
    #preprocess into object with users as keys
    #retains all information
    master = {}
    for jsonObj in data:
        if jsonObj['user_id'] in master:
            prefix = master[jsonObj['user_id']]['data']
            prefix[0].append(jsonObj['sentiment-google'])
            prefix[1].append(jsonObj['magnitude-google'])
        else:
            master[jsonObj['user_id']] = {'data':[
                [jsonObj['sentiment-google']],
                [jsonObj['magnitude-google']]
            ]
    }
    #add stats to each object
    for key in master:
        prefix = master[key]
        prefix['numberReviews'] = len(prefix['data'][0])
        prefix['averages'] = dictGOOG(calcAvg(prefix['data']))
        if len(prefix['data'][0]) > 1: prefix['stdev'] = dictGOOG(calcStdev(prefix['data']))
        else: prefix['stdev'] = {'score': 0, 'magnitude': 0}
        prefix['median'] = dictGOOG(calcMed(prefix['data']))
        prefix['max'] = dictGOOG(calcMax(prefix['data']))
        prefix['min'] = dictGOOG(calcMin(prefix['data']))
    return master
    
# =============================================================================
# GROUP BY BUSINESS SUBMASTER FUNCTION
# =============================================================================
def busGroupNLTK(data):
    #preprocess into object with users as keys
    #retains all information
    master = {}
    for jsonObj in data:
        if jsonObj['business_id'] in master:
            prefix = master[jsonObj['business_id']]['data']
            prefix[0].append(jsonObj['negative'])
            prefix[1].append(jsonObj['neutral'])
            prefix[2].append(jsonObj['positive'])
            prefix[3].append(jsonObj['compound'])
        else:
            master[jsonObj['business_id']] = {'data':[
                [jsonObj['negative']],
                [jsonObj['neutral']],
                [jsonObj['positive']],
                [jsonObj['compound']]
            ]
    }
    #add stats to each object
    for key in master:
        prefix = master[key]
        prefix['numberReviews'] = len(prefix['data'][0])
        prefix['averages'] = dictNLTK(calcAvg(prefix['data']))
        if len(prefix['data'][0]) > 1: prefix['stdev'] = dictNLTK(calcStdev(prefix['data']))
        else: prefix['stdev'] = {'negative': 0, 'neutral': 0, 'positive': 0, 'compound': 0}
        prefix['median'] = dictNLTK(calcMed(prefix['data']))
        prefix['max'] = dictNLTK(calcMax(prefix['data']))
        prefix['min'] = dictNLTK(calcMin(prefix['data']))
    return master
def busGroupGOOG(data):
    #preprocess into object with users as keys
    #retains all information
    master = {}
    for jsonObj in data:
        if jsonObj['business_id'] in master:
            prefix = master[jsonObj['business_id']]['data']
            prefix[0].append(jsonObj['sentiment-google'])
            prefix[1].append(jsonObj['magnitude-google'])
        else:
            master[jsonObj['business_id']] = {'data':[
                [jsonObj['sentiment-google']],
                [jsonObj['magnitude-google']]
            ]
    }
    #add stats to each object
    for key in master:
        prefix = master[key]
        prefix['numberReviews'] = len(prefix['data'][0])
        prefix['averages'] = dictGOOG(calcAvg(prefix['data']))
        if len(prefix['data'][0]) > 1: prefix['stdev'] = dictGOOG(calcStdev(prefix['data']))
        else: prefix['stdev'] = {'score': 0, 'magnitude': 0}
        prefix['median'] = dictGOOG(calcMed(prefix['data']))
        prefix['max'] = dictGOOG(calcMax(prefix['data']))
        prefix['min'] = dictGOOG(calcMin(prefix['data']))
    return master
   
# =============================================================================
# OUTPUT DICT HELPERS
# =============================================================================
def dictNLTK(sentimentList):
    item = {}
    item['negative'] = sentimentList[0]
    item['neutral'] = sentimentList[1]
    item['positive'] = sentimentList[2]
    item['compound'] = sentimentList[3]
    return item
def dictGOOG(sentimentList):
    item = {}
    item['score'] = sentimentList[0]
    item['magnitude'] = sentimentList[1]
    return item

# =============================================================================
# OUTPUT FILE CREATOR
# =============================================================================
def outputFile(data, name):
    with open(name, 'w') as outfile:
        json.dump(data, outfile,indent=4)

# =============================================================================
# MASTER FUNCTION TO PROCESS DATA
# =============================================================================
def processNLTK(data):
    outputFile(generalNLTK(data), 'general-nltk-sentiment.json')
    outputFile(userGroupNLTK(data), 'user-nltk-sentiment.json')
    outputFile(busGroupNLTK(data), 'business-nltk-sentiment.json')
def processGOOG(data):
    outputFile(generalGOOG(data), 'general-google-sentiment.json')
    outputFile(userGroupGOOG(data), 'user-google-sentiment.json')
    outputFile(busGroupGOOG(data), 'business-google-sentiment.json')

# =============================================================================
# MAIN METHOD
# =============================================================================
if __name__ == '__main__':
    #open and load files
    #open file
    with open('nltk_results.json', 'r') as f:
        nltk_data = json.load(f)
    with open('google_results.json', 'r') as g:
        google_data = json.load(g)
        
    processNLTK(nltk_data)
    processGOOG(google_data)