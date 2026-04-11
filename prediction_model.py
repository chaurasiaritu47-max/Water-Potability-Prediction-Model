import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


df = pd.read_csv('water_potability.csv')
df.head()

df.shape

df.rename(columns={'Sulfate': 'Sulphate'}, inplace=True)
df.head()

df.info()

plt.figure(figsize=(15,5))    #(15,5) = 15 is for horizontal(length) and 5 is for vertical(breadth)
sns.barplot(df.isnull())

df["ph"]= df["ph"].fillna(df["ph"].mean())
df["Sulphate"]= df["Sulphate"].fillna(df["Sulphate"].mean())
df["Trihalomethanes"]= df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

sns.heatmap(df.isnull())

df.isnull()

df.describe()

plt.figure(figsize=(10,8))  #figure will give size to the heatmap
sns.heatmap(df.corr(),annot=True)

df.Potability.value_counts()   #0= unsafe and 1=safe

sns.countplot(x= "Potability",data=df,hue="Potability" )

sns.displot(df['ph'])    #this show the range of ph value high in dataset
plt.show()

df.hist(figsize=(12,12))
plt.show()

sns.scatterplot(df, x="ph", y="Hardness" ,hue='Potability', palette='coolwarm')

df.boxplot(figsize=(15,7))   #shows outliers

X = df.drop('Potability', axis=1)  # Features
y = df['Potability']               # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=181, shuffle=True)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

dtree= DecisionTreeClassifier(criterion='gini', min_samples_split =10, splitter ='best')
dtree.fit(X_train, y_train)

pred_y = dtree.predict(X_test)
print(f"Accuracy Score ={accuracy_score(y_test,pred_y)*100}")
print(f"Confusion matrix =\n{confusion_matrix(y_test,pred_y)}")
print(f"classification Report=\n{classification_report(y_test, pred_y)}")

columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulphate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

input_data = pd.DataFrame([[9.092223,	181.101509,	17978.986339,	6.546600,	310.135738,	398.410813,	11.558279,	31.997993,	4.075075]], columns=columns)  #for 0(notsafe)
input_data1 = pd.DataFrame([[8.923981,132.8328, 11557.03 ,8.550573 ,323.5081 ,442.768 ,13.37615 ,92.17617 ,5.133478]], columns=columns) #for 1(safe)
input_data2=pd.DataFrame([[10.04103,113.8311,16266.43,7.939074,363.8669,557.4861,13.19534,75.23322,3.807563]],columns=columns) #for 1
input_data3=pd.DataFrame([[10.02616,224.2664,34864.43,8.951692,385.5056,324.5098,12.69654,97.11286,4.592075]], columns=columns) #for 0
input_data4=pd.DataFrame([[6.79,187.78,15300.84,7.48,316.95,498.66,19.60,55.40,5.16]],columns=columns)  #for 1
input_data5=pd.DataFrame([[1.29,235.73,36043.71,5.19,377.19,385.61,17.05,89.62,4.16]],columns=columns)  #for 1
input_data6=pd.DataFrame([[8.07,195.80,29483.6,10.49,298.73,321.74,9.75,72.73,3.68]])   #for 1
#making prediction
res = dtree.predict(input_data6)[0]
res
#res1

#importing the model
joblib.dump(dtree, 'water_purity_model.pkl')

