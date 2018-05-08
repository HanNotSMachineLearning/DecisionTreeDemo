import csv
from random import shuffle
from sklearn import tree
import time

# read csv files
with open('FruitData.csv', 'r') as DataFile:
	csvList = list(csv.reader(DataFile))
	shuffle(csvList)
	trainDataCount = round(len(csvList) * 7 / 10)+1
	trainData = csvList[0:trainDataCount].copy()
	testData = csvList[trainDataCount:].copy()
	
with open('FruitNames.csv', 'r') as NamesFile:
	fruitNames = list(csv.reader(NamesFile))
	
# datasets
test_features = []
test_labels = []
	
features = []
labels = []

# split labels from features
for item in testData:
	test_labels.append(item[0])
	test_features.append(item[1:4].copy())
	
for item in trainData:
	labels.append(item[0])
	features.append(item[1:4].copy())
	
# create a decision tree classifier
DT_clf = tree.DecisionTreeClassifier()
# train the classifier with the trainingsdata
DT_clf = DT_clf.fit(features,labels)

print("\nHello, I am Fruity. \nI can predict what kind of fruit you have. \nFor this I only need to know the height, width and weight.")
print("\nPlease only ask for pieces of fruit that I know, these are the following:")
for fruit in fruitNames:
	print("\t- "+fruit[1])

while True:
	print("\nTell me what the height of your piece of fruit is in cm.")
	#get height value from user
	height = input("")
	print("\nWhat is the width of the piece of fruit in cm?")
	#get width value from user
	width = input("")
	print("\nAnd finally what is the weight in grams?")
	#get weight value from user
	weight = input("")
	#predict fruit type on the basis of data entered by the user
	prediction = DT_clf.predict([[weight,width,height]])[0]
	#link correct fruitname to predicted value
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