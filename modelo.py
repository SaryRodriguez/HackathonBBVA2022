# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:50:53 2022

@author: rodrigsa
"""

#%%Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score,precision_score,recall_score,f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

#%%Cargar datos
data = pd.read_excel('datos.xlsx')

#%% Filtrar por los que no continuaron con el proceso
df2 = data[data['Continuar_proceso']=='SI']
df1 = data[data['Continuar_proceso']=='NO']

#%%#Limpieza

df= df1.copy()

#Datos Nulos
df.isnull().sum()
#Rellenar datos nulos
df['Producto_anterior'] = df['Producto_anterior'].fillna("NA")

#Tipos de datos
df.info()

#Convertir datos categóricos a numericos
df["Es_activo"] = df["Es_activo"].replace(["Si","No"],[1,0]).astype(float)
df["Es_lead"] = df["Es_lead"].astype(float)


#convertir datos categóricos a dummies
cat_col=[ 'Genero', 'Ocupacion','Codigo_canal', 'Producto_anterior']
le = LabelEncoder()     
for col in cat_col:
  df[col]= le.fit_transform(df[col])
df_2=df


#%%Seleccionar columnas a trabajar en el modelo

df_2 = df_2[['Genero', 'Edad', 'Ocupacion', 'Producto_anterior', 'Saldo_promedio_cuenta', 'Es_activo', 'Antiguedad',
       'Codigo_canal', 'Es_lead']]

#%%

#Edad
sns.displot(df_2, x="Edad")
#Los datos están sesgados a la derecha. Un grupo tiene entre 20 y 40 años y otro grupo entre 40 y 60

#Genero
sns.countplot(x='Genero',data=df_2)
#Existen más hombres

#Ocupacion por clientes potenciales
sns.countplot(x='Ocupacion', hue='Es_lead', data=df_2)

#Ocupacion por clientes activos
sns.countplot(x='Ocupacion', hue='Es_activo', data=df_2)

#VAriable target <- Es lead
sns.countplot(x='Es_lead',data=df_2)
#Datos imbalanceados


#%%Preparar datos

#Separar X y Y
X = df_2.drop('Es_lead', axis=1)
y= df_2[['Es_lead']]

#Separar train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Rebalancear datos
smote_nc = SMOTENC(categorical_features=[0, 2, 3, 5, 7], random_state=0)
X_train, y_train = smote_nc.fit_resample(X_train, y_train)

#Escalar
sc = StandardScaler()        
X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X.columns)
X_test = pd.DataFrame(sc.fit_transform(X_test),columns = X.columns)


#%%Modelo

#Cross Validation
n_folds = 5
seed = 7
kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)


#%%Logistic Regression
#Inicializar modelo
log_model2=LogisticRegression(max_iter=10000)
log_acc2 = cross_val_score(log_model2, X_train, y_train, scoring='accuracy', cv=kfold)
log_prec2 = cross_val_score(log_model2, X_train, y_train, scoring='precision', cv=kfold)
log_rec2 = cross_val_score(log_model2, X_train, y_train, scoring='recall', cv=kfold)
log_roc2 = cross_val_score(log_model2, X_train, y_train, scoring='roc_auc', cv=kfold)

#Predicting Test set
log_model2.fit(X_train, y_train)
y_pred_log2 = log_model2.predict(X_test)
y_pred_log_prob2 = log_model2.predict_proba(X_test)

acc = accuracy_score(y_test,y_pred_log2)
prec = precision_score(y_test, y_pred_log2)
rec = recall_score(y_test, y_pred_log2)
f1 = f1_score(y_test,y_pred_log2)
roc_auc = roc_auc_score(y_test, y_pred_log2)
results = pd.DataFrame([['Logistic Regression (Lasso)', acc,prec,rec,f1,roc_auc]],columns=['Model', 'Accuracy', 'Precision', 'Recall','F1 Score','ROC AUC'])
print(results)


#%%Random Forest

#rf_clf=RandomForestClassifier()
#parameters={"n_estimators":[1,10,100], 'max_depth': [4,8,10]}
#clf = GridSearchCV(rf_clf, parameters, cv=kfold,scoring="roc_auc")

#Predicting Test set
#clf.fit(X_train, y_train)
#print("Best parameter : ",clf.best_params_)
#Best parameter :  {'max_depth': 10, 'n_estimators': 100}

rf_clf=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf2 = rf_clf.predict(X_test)
y_pred_rf_prob2 = rf_clf.predict_proba(X_test)

acc = accuracy_score(y_test,y_pred_rf2)
prec = precision_score(y_test, y_pred_rf2)
rec = recall_score(y_test, y_pred_rf2)
f1 = f1_score(y_test,y_pred_rf2)
roc_auc = roc_auc_score(y_test, y_pred_rf2)
results2 = pd.DataFrame([['RF', acc,prec,rec,f1,roc_auc]],columns=['Model', 'Accuracy', 'Precision', 'Recall','F1 Score','ROC AUC'])
print(results2)

#%%Guardar modelo
import pickle
filename = 'C:\\Users\\rodrigsa\\OneDrive - HP Inc\\Hackathon\\finalized_model.sav'
pickle.dump(rf_clf, open(filename, 'wb'))


#%%Graficar densidades

graph_predictions2 = pd.DataFrame(y_pred_rf_prob2, columns=['proba_false','proba_true'])
graph_predictions_class2 = pd.DataFrame(y_pred_rf2, columns=['prediction'])
real2 = y_test
gr2 = pd.concat([real2, graph_predictions2.set_index(real2.index[:len(graph_predictions2)]), graph_predictions_class2.set_index(real2.index[:len(graph_predictions_class2)])], axis=1)

gr2.head()
sns.kdeplot(data=gr2, x="proba_true", hue="Es_lead", 
            common_norm=False, bw_method=0.15)

#high -> prob >0.8
#medium -> 0.45 - 0.8
# Low -> prob < 0.45
#Unknown -> no prob

#%% assignar probabilidadeses
def probabilities(proba_true):
    if proba_true>=0.8:
        category='Alta'
    elif proba_true<0.8 and proba_true>=0.45:
        category='Media'
    else:
        category='Baja'
    return category

gr2['Propensidad'] = gr2.apply(lambda x: probabilities(x['proba_true']), axis=1)


#%%Añadir la variable de Asesor

final = pd.merge(df1, gr2, left_index=True, right_index=True)
final.drop(['prediction','proba_false', 'proba_true','Es_lead_x', 'Es_lead_y'],axis=1, inplace=True)

final['Asesor'] = ""

#%%añadir los que sí continuaron con el proceso
final1 = pd.concat([final, df2],ignore_index=True, axis=0)
final1.drop('Es_lead', inplace=True, axis=1)

#%%Exportar datos

final1.to_excel('C:\\Users\\rodrigsa\\OneDrive - HP Inc\\Hackathon\\datos_pred.xlsx')