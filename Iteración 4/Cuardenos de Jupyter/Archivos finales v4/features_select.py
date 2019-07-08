#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.metrics import v_measure_score
import metrics
class MIFS:
    def __init__(self,beta = 0.5):
        self.beta = beta
        self.dataset = None
        self.column_y = ''
        self.m_im = None
        self.features  = []
        
    def fit(self,X,y, dataset = None, column_y=''):
        if dataset:
            self.dataset = pd.concat([X,y],axis=1)
            self.column_y = column_y
            self.__fit()
        else:
            self.dataset = pd.concat([X,y],axis=1)
            self.column_y = y.name
            self.__fit()
            
            
            
    def __fit(self):
        matriz = []
        for i in self.dataset.columns:
            x = list(self.dataset[i])
            print(i)
            fila = []
            for j in self.dataset.columns:
                y = list(self.dataset[j])
                fila.append(v_measure_score(x,y))
            matriz.append(fila)
    

        m_im = pd.DataFrame(matriz,columns=self.dataset.columns,index=self.dataset.columns)
        self.m_im = m_im
        self.__mifs(m_im)
    
                  
    def __mifs(self,matriz_informacionmuta):
        
        im = 0
        features_candidate = list(self.dataset.columns)
        features_candidate.pop(features_candidate.index(self.column_y))
        features = []
        #features_candidate = columnas
        best_candidate = ['',0.00000000001]
        
        while best_candidate[1] > im:
            best_candidate = ['',0]
            flag = False
            #print(features)
            for i in features_candidate:
                im_p = im + matriz_informacionmuta.loc[i,self.column_y] - (self.beta * sum([matriz_informacionmuta.loc[i,j] for j in features]))
                
                if im_p > best_candidate[1]:
                    best_candidate[0] = i
                    best_candidate[1] = im_p
                    flag = True
                    
            if flag:
                val = features_candidate.pop(features_candidate.index(best_candidate[0]))
                features.append(val)
        self.features = features  
        #return features
    
    def transform(self,X):
        return X[self.features]
        
    
    def fit_transform(self,X,y,dataset=None,column_y = ''):
        self.fit(X,y,dataset = dataset, column_y = column_y)
        return X[self.features]


class busquedaWrapper():

    def __init__(self,X,y,model):
        self.X=X
        self.y=y
        self.model=model
        self.features=X.columns
        self.final_features=[]
        
    def fit(self):
        
        from sklearn.model_selection import cross_val_score
        
        new_features=[]
        features_candidate = list(self.features[:])
        best_candidate = ''
        bestlocalscore=0
        bestscore=-1
        
        
        while(bestlocalscore>bestscore):
            
            best_candidate = ''
            #print('hola')
            bestscore=bestlocalscore
            
            for feature in features_candidate:
                
                
                score=cross_val_score(self.model, self.X[new_features + [feature]], self.y, cv=10, scoring=metrics.fpr_score)
                
                print('score iteracion:'+str(score.mean())+',mejor score:'+str(bestlocalscore))
                
                if score.mean()>bestlocalscore:
                    bestlocalscore=score.mean()
                    best_candidate = feature
                    
            #print(features_candidate)
            if best_candidate != '':
                print(best_candidate)
                features_candidate.pop(features_candidate.index(best_candidate))
                new_features=new_features + [best_candidate]
            print(features_candidate)
            print('iteraccion terminada')
        self.features=new_features[:]
    
    def transform(self,X = 0):
        if X == 0:
            X = self.X
        return X[self.features]
    
    def fit_transform(self):
        self.fit()
        return self.transform()


def best_beta_mifs(lista_valores,X,y):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score

    X_best = None
    accuracys = []
    model = GaussianNB()
    best_score = 0
    best_beta = None
    
    for i in lista_valores:
        print(i)
        X_new = MIFS(beta = i).fit_transform(X,y.iloc[:,0])
        
        best_score_p = cross_val_score(model,X_new,y.iloc[:,0],cv=10,scoring=metrics.fpr_score).mean()
        accuracys.append(best_score_p)
        
        print(str(best_score)+': mejor vs nuevo :'+str(best_score_p))
        
        if best_score_p > best_score:
            best_beta = i
            X_best = X_new
            best_score = best_score_p
    
    return best_beta,X_best,accuracys

def new_columns(name,name2,X,new_column,sign):
    
    new_data = pd.DataFrame()
    positive = [name+' Últ. año disp.', name+' Año - 1',name+' Año - 2']
    neg = [name2+' Últ. año disp.', name2+' Año - 1',name2+' Año - 2']
    
    for i in range(len(positive)):
        if sign == '+':
            
            if 'Últ. año disp.' in positive[i]:
                new_data[new_column+' Últ. año disp.'] = X[positive[i]] + X[neg[i]]
            else:
                new_data[new_column+positive[i][-8:]] = X[positive[i]] + X[neg[i]]

                
        if sign == '-':
            
            if 'Últ. año disp.' in positive[i]:
                new_data[new_column+' Últ. año disp.'] =  X[positive[i]] - X[neg[i]]
            else:
                new_data[new_column+positive[i][-8:]] = X[positive[i]] - X[neg[i]]

                
        if sign == '/':
            
            if 'Últ. año disp.' in positive[i]:
                new_data[new_column+' Últ. año disp.'] = (X[positive[i]] / X[neg[i]])*100
                
                for j in range(len(new_data[new_column+' Últ. año disp.'])):
                    if new_data[new_column+' Últ. año disp.'].iloc[j] == np.inf:
                        new_data[new_column+' Últ. año disp.'].iloc[j] = X[positive[i]].iloc[j]
                            
            else:

                new_data[new_column+positive[i][-8:]] = (X[positive[i]] / X[neg[i]])*100
                
                for j in range(len(new_data[new_column+positive[i][-8:]])):
                    if new_data[new_column+positive[i][-8:]].iloc[j] == np.inf:
                        new_data[new_column+positive[i][-8:]].iloc[j] = X[positive[i]].iloc[j]
                
      

        if sign == '*':
            
            if 'Últ. año disp.' in positive[i]:
                new_data[new_column+' Últ. año disp.'] = X[positive[i]] * X[neg[i]]
            else:
                new_data[new_column+positive[i][-8:]] = X[positive[i]] * X[neg[i]]

    return new_data

def sort_columns(columns):
    columns1 = []
    columns3 = []
    for i in range(len(columns)):
        #print(columns[i])
        if 'Últ. año disp.' in columns[i]:
            #print('primer if')
            columns3 = columns3 + [columns[i],columns[i+1],columns[i+2]]
        elif not(('Últ. año disp.' in columns[i]) or ('Año - 1' in columns[i]) or ('Año - 2' in columns[i] )):
            #print('segundo')
            columns1 = columns1 + [columns[i]]
            
    return columns1 + columns3

def generate_datasets(X):
    ##Crear array por año
    names = (X['Código NIF'].values).reshape((X.shape[0],1))
    features_extra = (X[['Tamano']].values).reshape((X.shape[0],1))
    X_ult = names
    X_ano1 = names
    X_ano2 = names
    print(len(X.columns))
    for i in range(2,len(X.columns)):
        resto = (i-2) % 3
        print('i:'+str(i))
        print('resto:'+str(resto))
        if resto == 0:
            #print(1)
            print(X.columns[i])
            X_ult = np.append(X_ult,(X[X.columns[i]].values).reshape((X.shape[0],1)),axis=1)
            
        elif resto == 1:
            #print(2)
            print(X.columns[i])
            X_ano1 = np.append(X_ano1,(X[X.columns[i]].values).reshape((X.shape[0],1)),axis=1)
        
        elif resto == 2:
            #print(3)
            print(X.columns[i])
            X_ano2 = np.append(X_ano2,(X[X.columns[i]].values).reshape((X.shape[0],1)),axis=1)
    
    X_ult = np.concatenate((features_extra,X_ult),axis=1)
    print(X_ult[0])
    
    X_ano1 = np.concatenate((features_extra,X_ano1),axis=1)
    print(X_ano1.shape)
    
    X_ano2 = np.concatenate((features_extra,X_ano2),axis=1)
    print(X_ano2.shape)
    
    x  = np.concatenate(([X_ano2[0,:]],[X_ano1[0,:]],[X_ult[0,:]]),axis=0)
    
    for j in range(1,X.shape[0]):
        print(j)
        x = np.concatenate((x,[X_ano2[j,:]],[X_ano1[j,:]],[X_ult[j,:]]),axis=0)
        
    x =  np.delete(x,[1],axis=1)
    
    return x.reshape(X.shape[0],3,x.shape[0])
                       