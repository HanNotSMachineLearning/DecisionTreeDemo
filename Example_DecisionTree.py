import csv
from random import shuffle
from sklearn import tree
import time

with open('FruitData.csv', 'r') as DataFile:
	csvList = list(csv.reader(DataFile))
	shuffle(csvList)
	trainDataCount = round(len(csvList) * 7 / 10)+1
	trainData = csvList[0:trainDataCount].copy()
	testData = csvList[trainDataCount:].copy()
	
with open('FruitNames.csv', 'r') as NamesFile:
	fruitNames = list(csv.reader(NamesFile))
	
test_features = []
test_labels = []
	
features = []
labels = []

for item in testData:
	test_labels.append(item[0])
	test_features.append(item[1:4].copy())
	
for item in trainData:
	labels.append(item[0])
	features.append(item[1:4].copy())
	
DT_clf = tree.DecisionTreeClassifier()
DT_clf = DT_clf.fit(features,labels)

print("\nHello, I am Fruity. \nI can predict what kind of fruit you have. \nFor this I only need to know the height, width and weight.")

while True:
	print("\nTell me what the height of your piece of fruit is in cm.")
	height = input("")
	print("\nWhat is the width of the piece of fruit in cm?")
	width = input("")
	print("\nAnd finally what is the weight in grams?")
	weight = input("")
	prediction = DT_clf.predict([[weight,width,height]])[0]
	for name in fruitNames:
		if name[0] == prediction:
			print("\n\nI think you have a:")
			print("\t\t"+name[1])
			break
	time.sleep(1)
	print("\n\n\nDo you want me to predict a piece of fruit again? (Y/N)")
	again = input("")
	if again.upper() != "Y":
		break

print("\nOkay you see the next time!")