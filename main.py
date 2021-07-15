import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv("voice.csv")

#converting male female to 1 and 0
data.label[data.label=="male"]=1
data.label[data.label=="female"]=0

# x as features and y as labels
x= data.iloc[:,:20]
y= data["label"]
y = y.to_frame()        #converting y from list to dataframe


r=[]
for i in range(1500,1584):
    r.append(i)

#training feature     
train_f = x.iloc[:3100,:]
train_f = train_f.drop(r)
train_f = train_f.as_matrix()

#training label    
train_l=  y.iloc[:3100,:]
train_l= train_l.drop(r)     
train_l = train_l.as_matrix()


#testing feature
df1 = x.iloc[r]
df2 = x.iloc[3100:,:]
test_f = pd.concat([df1,df2])
test_f = test_f.as_matrix()


#testing labels
df3 =y.iloc[r]
df4 = y.iloc[3100:,:]
test_l = pd.concat([df3,df4])
test_l = test_l.as_matrix()

test_l=test_l.astype('float64')
train_l=train_l.astype('float64')


# using ANN

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#initialising  the ANN
classifier = Sequential()

# adding input layer and hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform',activation='relu',input_dim = 20))
#adding second hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform',activation='relu',))
#adding third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform',activation='relu',))


#adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation='sigmoid',))

#compiling ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
#fitting the ANN to training  set
classifier.fit(train_f,train_l,batch_size=10,epochs=180)
#making prediction and evaluating the model
pred = classifier.predict(test_f)
#converting probablity of pred in 0 or 1
pred = (pred>0.5)  #in true or false

#checking accuracy
cm = confusion_matrix(pred,test_l)
print(cm)
acc=accuracy_score(pred,test_l)
print(acc*100)