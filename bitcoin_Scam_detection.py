import pandas as pd
import matplotlib.pyplot as plt

def highest_vol(df):
print("\nPrice With Highest Volume")
return df.loc[df['Volume_BTC'].idxmax()]

def VWAP_vs_volume(df):
plt.title("Volume vs Weighted Price")
plt.plot(df.Weighted_Price.iloc[::500], linewidth=1, color='blue', label='WP')
plt.plot(df.Volume_BTC.iloc[::500], linewidth=1, color='red', label='Volume')

def price(df):
plt.cla()
plt.title("Price")
plt.plot(df.High, linewidth=1, color='blue', label='Price')

def price_vs_volume_high(df):
mean = df.High.mean()
plt.figure(figsize=(8,5))
plt.title("Volume vs Prices Above Mean")
plt.plot(df.High[df.High>mean].iloc[::500], linewidth=2, label='Price')
plt.plot(df.Volume_BTC.iloc[::500], linewidth=1, color='red', label='Volume')

44

def price_vs_volume_low(df):
mean = df.High.mean()
plt.figure(figsize=(8,5))
plt.title("Prices Below Mean")
plt.plot(df.High[df.High<mean].iloc[::500], linewidth=2, label='Price')

def volume_stats(df):
print("BTC Volume Statictics...")
return df.Volume_BTC.describe()

def high_vs_vol(df):
plt.figure(figsize=(8,5))
plt.plot(df.High.iloc[::500], linewidth=2, label='Price')
plt.plot(df.Volume_BTC.iloc[::500], linewidth=1, color='red', label='Volume')
plt.title('Price vs Volume')
plt.show()

def main():
file = "new_database.csv" df = pd.read_csv(file)

high_vs_vol(df)

print(volume_stats(df))

price_vs_volume_low(df)

price_vs_volume_high(df)

45

VWAP_vs_volume(df)

print(highest_vol(df))

price(df)

if name == ' main ':
main()

import seaborn as sns
plt.rcParams["figure.figsize"] = (11,3)
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("new_database.csv", sep=",")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')df['Date'] = df['Timestamp'].dt.date

46
data_day=df.groupby("Date").mean()
data_day.rename(columns={'Volume_(BTC)': 'Volume_BTC',
'Volume_(Currency)': 'Volume_Currency'}, inplace=True)
fn=list(data_day.columns)
f,ax = plt.subplots(figsize=(9, 4))
sns.heatmap(data_day.corr(), annot=True, linewidths=5, fmt= '.1f',ax=ax)
plt.xticks(rotation= 45)
plt.show()
data_day.isnull().sum()
df=df.values

test_days=30
train=df[:-test_days] # the days before the last 30 days
test=df[len(df)-test_days:]
df=pd.DataFrame(data_day.Weighted_Price)
df.shape
#

df=pd.DataFrame(data_day.Weighted_Price)
df.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_s = scaler.fit_transform(df) # df_s : df_scaled
df_X, df_y = [],[]
first=10 # first array= time steps
last =len(df_s) # last array

47

for i in range(first,last):
df_X.append(df_s[i-first:i])
df_y.append(df_s[i])
import numpy as np

df_X, df_y = np.array(df_X), np.array(df_y)
test_days=30
X_train=df_X[:-test_days] # the days before the last 30 days
y_train=df_y[:-test_days]
X_test=df_X[len(df_X)-test_days:] # the last 30 days
y_test=df_y[len(df_X)-test_days:]
print(df.shape)
print(df.columns)

df_X = df_X.reshape(df_X.shape[0], -1)
df_y = df_y.reshape(df_y.shape[0], -1)
df_y=df_y.round()

x_train,x_test,y_train,y_test = train_test_split(df_X,df_y,test_size =
0.25,random_state = 42)

model = LogisticRegression(solver='liblinear', random_state=0)
clf=LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred=clf.predict(x_test)

Result_4=metrics.accuracy_score(y_test,(y_pred))*100
print()
print(" ")

48
print("LogisticRegression ")
print()
print("LogisticRegression Acuracy is :",Result_4,'%')
print(metrics.classification_report(y_test , (y_pred)))
print("Confusion Matrix:")
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(" ")
print()
import matplotlib.pyplot as plt
plt.imshow(cm, cmap='binary')
import seaborn as sns
sns.heatmap(cm, annot = True, cmap ='plasma', linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, marker='.', label='LogisticRegression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#

from sklearn.ensemble import RandomForestClassifier

49

rf= RandomForestClassifier(n_estimators = 10)
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print(" ")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print(" ")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma', linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, rf_prediction)

50
plt.plot(fpr, tpr, marker='.', label='RF')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#importsmtplib, ssl

inp=int(input('Enter the Scam '))

if (rf_prediction[inp]==True):
if (rf_prediction[inp] ==0 ):
print("BLACKMAIL ")
elif (rf_prediction[inp] ==1 ):
print("FAKEICO ")
else:
print("FAKESERVICE")

print("Scam detected ")
file2 = open(r"C:\Users\divyadeepak\CloudMe\output.txt","w+")
file2.write("SCAM detected --https://www.bitcoinabuse.com/")
file2.close()

else :
print("Scam Not detected")