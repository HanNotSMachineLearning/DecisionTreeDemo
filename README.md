# Machine learning decision tree in Python

Deze tutorial laat zien hoe je een decision tree machine learning applicatie kan maken in Python.

Wil je niet de hele tutorial volgen maar alleen het eind resultaat zien klik dan [hier](##Programma draaien).



## Afhankelijkheden

Deze demo is geschreven en getest binnen een Windows 10 omgeving en niet getest op andere platforms. Het is dus niet gegarandeerd dat deze demo ook werkt op andere platforms als MacOS en Linux.

Voor deze demo dienen een aantal zaken geïnstalleerd te zijn op de omgeving waarin deze demo wordt uitgevoerd.

- Python versie 3.6 	(https://www.python.org/downloads/release/python-364/)
- Pip versie 10.0  		(https://pip.pypa.io/en/stable/installing/)




## Tutorial

Aan de hand van het onderstaande stappenplan gaan we een Python applicatie die gebruikt maakt van het decision tree machine learning algoritme.

### 1. Project opzetten

Voordat we kunnen beginnen aan het maken van de applicatie moeten we de omgeving inrichten waarin we de applicatie gaan bouwen.

1. Zorg dat je alle afhankelijkheden geïnstalleerd hebt.
2. Maak een folder aan voor dit project. Hierin gaan we alles van deze tutorial bewaren.
2. Kopieer de twee CSV-bestanden uit dit repository naar de net gemaakte folder. Deze bestanden bevatten de data om het machine learning algoritme te trainen.
4. Open een PowerShell-venster en navigeer naar de net aangemaakte folder.
5. Run het volgende commando `pip install -U scikit-learn` in het net geopende PowerShell-venster. Hiermee installeren we scikit. Dit framework gaan we gebruiken voor onze machine learning applicatie.
5. In de projectfolder, maak een nieuw bestand aan genaamd `Example_DecisionTree.py`. Dit is het bestand voor ons programma.



###2. Python programma bouwen

Nu we een omgeving hebben voor onze applicatie kunnen we beginnen met het ontwikkelen van onze applicatie.

1. Open het zojuist aangemaakte `Example_DecisionTree.py` bestand.

2. Als eerste gaan we er voor zorgen dat we data voor in onze applicatie hebben om het machine learning model te trainen. Dit gaan we doen door de twee CSV-bestanden in te lezen in de applicatie. Hiervoor moeten we als eerste de csv module van Python importeren. Daarna kunnen we de twee CSV-bestanden inlezen zoals hieronder. Controleer of de code werkt door het commando `python Example_DecisionTree.py` uit te voeren in het PowerShell-venster, als het goed is krijg je twee lijsten met data te zien.

	```python
	import csv

	with open('FruitData.csv', 'r') as DataFile:
		fruitData = list(csv.reader(DataFile))
		print(fruitData)
	
	with open('FruitNames.csv', 'r') as NamesFile:
		fruitNames = list(csv.reader(NamesFile))
		print(fruitNames)
	```

3. Nu gaan we de ingelezen data geschikt maken om te gebruiken voor het trainen van het machine learning algoritme. Als eerste gaan we de data opsplitsen in trainings- en testdata. Hiervoor moeten we de ingeladen data eerst husselen. Dit doen we door de shuffle methode te gebruiken. Vervolgens verdelen we de data volgens een 70/30 verhouding over de trainings- en testdata. Hierbij scheiden we ook meteen het label van de features omdat dit moet voor het scikit-framework. Onze code ziet er nu als volgt uit:

   ```python
   import csv
   from random import shuffle

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
   ```

4. Nu onze data geschikt is om te gebruiken kunnen we het machine learning gedeelte van onze applicatie gaan bouwen. Er zijn twee dingen die we moeten doen. Als eerste moeten we een classifier aanmaken. Deze moeten we vervolgens trainen door middel van de trainingsdata.

   ```python
   import csv
   from random import shuffle
   from sklearn import tree

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

   # test the classifier the loaded testdata
   print(DT_clf.predict(test_features))
   print(test_labels)
   ```
   Nu kunnen we door middel van de test data controleren of onze applicatie werkt. Draai het programma weer via PowerShell en bekijk het resultaat. Als het goed is krijg je twee lijsten van getallen te zien, zoals hieronder weergeven. De eerste lijst zijn de voorspellingen van het machine learning algoritme, de tweede zijn de daadwerkelijke waardes. 

   `['3' '7' '8' '3' '2' '5' '8' '7' '5' '7' '6' '10' '10' '10' '8' '9' '8']`
   `['4', '7', '3', '5', '2', '5', '4', '5', '5', '7', '6', '10', '10', '10', '7', '7', '8']`

   ​

5. Als laatste gaan we nog wat simpele user controls toevoegen aan de applicatie om het zo bruikbaarder te maken.

   ```python
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
   	print("- "+fruit[1])

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
   ```

6. En we hebben een werkende machine learning applicatie.🎉🎉🎉

   ​


## Programma draaien

Geen zin om de tutorial te volgen maar wil je wel de applicatie proberen. Volg dan de volgende stappen.

1. Zorg dat je al de dependecies geïnstalleerd hebt.
2. Download of clone dit repository.
3. Open een powershell window en navigeer naar het repository.
4. Run het volgende commando `pip install -U scikit-learn`.
5. Run de applicatie d.m.v. het volgende commando `python Example_DecisionTree.py`.