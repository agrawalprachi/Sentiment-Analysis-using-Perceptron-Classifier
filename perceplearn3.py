import re
import sys
import json

#infile = sys.argv[1]

infile = "train-labeled.txt"
count = 1
bias_c1 = 0
bias_c2 = 0
beta_c1 = 0
beta_c2 = 0
vocabulary = set()
weight_c1 = dict()
weight_c2 = dict()
cweight_c1 = dict()
cweight_c2 = dict()
sen_word_freq_list = []


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
              'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
              'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
              'until', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
              'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
              'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
              'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
              'us', 'much', 'would', 'either', 'indeed', 'seems']


with open(infile) as f:
    for line in f:
        line  = line.strip()
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split(" ")

        class1 = words[1].strip()
        class2 = words[2].strip()

        temp_dict = {}

        for word in words[3:]:
            word = word.lower().strip()
            if word in stop_words:
                continue
            if word not in temp_dict:
                temp_dict[word] = 1
            else:
                temp_dict[word] += 1

        sen_word_freq_list.append((class1,class2,temp_dict))


for i in range(0,30):
    for item in sen_word_freq_list:
        c1 = item[0]
        c2 = item[1]
        sentence = item[2]
        activation1 = 0
        activation2 = 0
        for word, freq in sentence.items():
            if word in weight_c1:
                activation1 += weight_c1[word]*freq
            if word in weight_c2:
                activation2 += weight_c2[word]*freq
        activation1 += bias_c1
        activation2 += bias_c2

        if c1 == "True":
            y1 = 1
        else:
            y1 = -1

        if c2 == "Pos":
            y2 = 1
        else:
            y2 = -1

        if y1 * activation1 <= 0:
            bias_c1 += y1
            beta_c1 += y1*count
            for word,freq in sentence.items():

                if word in weight_c1:
                    weight_c1[word] += freq*y1
                else:
                    weight_c1[word] = freq*y1

                if word in cweight_c1:
                    cweight_c1[word] += freq*y1*count
                else:
                    cweight_c1[word] = freq*y1*count

        if y2 * activation2 <= 0:
            bias_c2 += y2
            beta_c2 += y2 * count
            for word, freq in sentence.items():

                if word in weight_c2:
                    weight_c2[word] += freq * y2
                else:
                    weight_c2[word] = freq * y2

                if word in cweight_c2:
                    cweight_c2[word] += freq * y2 *count
                else:
                    cweight_c2[word] = freq * y2*count
        count += 1

vanilla_data = {
            "bias_c1": bias_c1,
            "bias_c2": bias_c2,
            "weight_c1" : weight_c1,
            "weight_c2" : weight_c2
        }

averaged_data = {
            "count": count,
            "bias_c1": bias_c1,
            "bias_c2": bias_c2,
            "beta_c1": beta_c1,
            "beta_c2": beta_c2,
            "weight_c1" : weight_c1,
            "weight_c2" : weight_c2,
            "cweight_c1" : cweight_c1,
            "cweight_c2" : cweight_c2
        }

json1 = json.dumps(vanilla_data)
json2 = json.dumps(averaged_data)

f = open("vanillamodel.txt","w")
f.write(json1)
f.close()

f = open("averagedmodel.txt","w")
f.write(json2)
f.close()