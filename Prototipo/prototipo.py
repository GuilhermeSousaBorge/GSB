import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

# Home,Away,HG,AG,Res,PH,PD,PA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA
base = pd.read_csv('BRA.csv', encoding='ISO-8859-1')
base = base.drop('Country', axis=1)
base = base.drop('League', axis=1)
base = base.drop('Season', axis=1)
base = base.drop('Date', axis=1)
base = base.drop('Time', axis=1)

previsores = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]].values

labelencoder_previsores = LabelEncoder()
labelencoder_classe = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

classe = base.iloc[:, 4].values
classe = np.vstack(classe)
classe = labelencoder_classe.fit_transform(classe)

classeDummy = pd.get_dummies(classe)

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=7, activation='relu', input_dim=79))
    classificador.add(Dense(units=6, activation='relu'))
    classificador.add(Dense(units=5, activation='relu'))
    classificador.add(Dense(units=5, activation='relu'))
    classificador.add(Dense(units=4, activation='tanh'))
    classificador.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 10)

resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()

desvio = resultados.std()

print("Media de resultados = {:.2f}" .format(media*100))
print("Media de desvio = {:.2f}" .format(desvio*100))
