import sys
import json
import re

model = sys.argv[1]
file = sys.argv[2]

# model = "vanillamodel.txt"
# file = "dev-text.txt"
count = 0
bias_c1 = 0
bias_c2 = 0
beta_c1 = 0
beta_c2 = 0
weight_c1 = {}
weight_c2 = {}
cweight_c1 = {}
cweight_c2 = {}

fout = open("percepoutput.txt","w")

filename = model.rsplit("/", 1)
filename = filename[len(filename)-1]
if filename == "averagedmodel.txt":
    averaged_data = json.load(open('averagedmodel.txt'))
    count = averaged_data["count"]
    bias_c1 =  averaged_data["bias_c1"]
    bias_c2 =  averaged_data["bias_c2"]
    beta_c1 = averaged_data["beta_c1"]
    beta_c2 = averaged_data["beta_c2"]
    weight_c1 =  averaged_data["weight_c1"]
    weight_c2 = averaged_data["weight_c2"]
    cweight_c1 = averaged_data["cweight_c1"]
    cweight_c2 = averaged_data["cweight_c2"]

if filename == "vanillamodel.txt":
    vanilla_data = json.load(open('vanillamodel.txt'))
    bias_c1 = vanilla_data["bias_c1"]
    bias_c2 = vanilla_data["bias_c2"]
    weight_c1 = vanilla_data["weight_c1"]
    weight_c2 = vanilla_data["weight_c2"]

with open(file) as f:
    for line in f:
        line  = line.strip()
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split(" ")
        temp_dict = {}
        activation1 = 0
        activation2 = 0
        id = words[0].strip()
        for word in words[1:]:
            word = word.lower().strip()
            if word not in temp_dict:
                temp_dict[word] = 1
            else:
                temp_dict[word] += 1

        if filename == "vanillamodel.txt":
            for word,freq in temp_dict.items():
                if word in weight_c1:
                    activation1 += weight_c1[word]*freq
                if word in weight_c2:
                    activation2 += weight_c2[word]*freq
            activation1 += bias_c1
            activation2 += bias_c2

        elif filename == "averagedmodel.txt":
            for word, freq in temp_dict.items():
                if word in weight_c1 and word in cweight_c1:
                    activation1 += (weight_c1[word]-(cweight_c1[word]/float(count)))*freq
                if word in weight_c2 and word in cweight_c2:
                    activation2 += (weight_c2[word]-(cweight_c2[word]/float(count)))*freq
            activation1 += bias_c1 - beta_c1/float(count)
            activation2 += bias_c2 - beta_c2/float(count)

        class1 = ""
        class2 = ""

        if activation1 >= 0:
            class1 = "True"
        else:
            class1 = "Fake"

        if activation2 >=0:
            class2 = "Pos"
        else:
            class2 = "Neg"

        fout.write(id+ " "+ class1 + " " + class2 + "\n")
fout.close()