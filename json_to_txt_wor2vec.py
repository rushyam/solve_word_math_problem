import json
import re
file = open("train.txt","w+") 

with open('train.json') as json_file:  
    data = json.load(json_file)

for w in data:
    file.write(re.sub("[^ 0-9A-Za-z]", '', w["question"]).lower()+'\n')
file.close() 
