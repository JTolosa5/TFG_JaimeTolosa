#!/usr/bin/env python

##Montar Conjunto de datos.
from Imputer import trainNaN
import pandas as pd
import numpy as np

class mount_dataset():
    def __init__(self,dic,dic_dest,num):
        import pandas as pd
        self.dic = dic
        self.num = num
        self.dic_dest = dic_dest
        self.dataset=None


    def __mount_dataset(self,direccion,num):
        cont = 1
        df = pd.read_csv(direccion+str(cont)+'.txt',encoding = 'utf-16', index_col= False)
        cont+=1
        for i in range(cont,num+1):
            dfaux=pd.read_csv(direccion+str(i)+'.txt',encoding = 'utf-16',index_col = False)
            df = pd.concat([df,dfaux],ignore_index=True) # si no tomara la primera columna como el index
        
        return df        

    
    def fit(self):
        self.dataset = self.__mount_dataset(self.dic,self.num)
        self.dataset = self.dataset.drop(['Mark'],axis=1)
    
    def transform(self):
        #self.dataset = self.dataset.drop(['Mark'],axis=1)
        self.dataset.to_csv(self.dic_dest,index=False)



################################################################
############### PREPROCESAMIENTO ###############################
################################################################


class preprocesamiento():
    
    def __init__(self,dic_activas,dic_concurso):
        import numpy as np
        import pandas as pd

        self.activas = pd.read_csv(dic_activas,decimal=',')
        
        self.concursos = pd.read_csv(dic_concurso,decimal=',')


    
    def __new_columns(self):
        new_columns = []

        for i in self.concursos.columns:
            if i[-3:] == '- 2':
                new_columns.append(i[:-7]+'Últ. año disp.')
            elif i[-3:] == '- 3':
                new_columns.append(i[:-3]+'- 1')
            elif i[-3:] == '- 4':
                new_columns.append(i[:-3]+'- 2')
            else:
                new_columns.append(i)
                
        return new_columns
    
    def __delete_latest_year(self,dataset):
        dates = dataset['Ultimo año disponible']
        valores = []
        for date in dates:
            valores.append(int(date[6:]))

        date = pd.DataFrame(valores,columns=['Ult. año disponible'])
        dataset = pd.concat([dataset,date],axis=1)
        dataset =  dataset.drop(['Ultimo año disponible'],axis=1)
        return dataset

    def __transform_CNAE(self,dataset):

        letter=[]
        max_number=[]
        CNAE = pd.read_csv('../Datasets/CNAE/CNAE.csv').drop(['Unnamed: 0'],axis=1)

        for x,z in CNAE.groupby(['letter']):
            letter.append(x)
            max_number.append(max(z['number'].values))
            
        convert = list(zip(letter,max_number))
        CNAE_values=dataset['Código primario CNAE 2009'].values
        print(type(CNAE_values[1]))
        valors=[]
        for i in CNAE_values:
            flag = True
            for j in convert:
                if (np.isnan(i)) & (flag):
                    valors.append(np.nan)
                    flag=False
                elif (i <= j[1]) & (flag):
                    valors.append(j[0])
                    flag=False
        
        CNAE_number = pd.DataFrame({
        'CNAE_Number':valors
        })


        return pd.concat([dataset,CNAE_number],axis=1)

    def __dataset_to_float(self,dataset):
        for i in range(7,len(dataset.columns)-1):
            #print(i)
            #if dataset.dtypes[dataset.columns[i]] == 'O':
            dataset[dataset.columns[i]] = dataset[dataset.columns[i]].astype(np.float64)
        return dataset

    def __drop_columns_nan(self,dataset):
        num_nulos = int(dataset.shape[0] / 2)
        datos_nulos=dataset.isnull().sum()
        delete=datos_nulos[datos_nulos>=num_nulos]
        #print(delete)
        sup=list(delete.index)
        dataset = dataset.drop(sup,axis=1)
        return dataset

    def __dummis_variable_Formajuridica(self,data):

        formas = pd.get_dummies(data['Forma jurídica'])
        cols = list(set(data['Forma jurídica'].value_counts().index) - set(['Sociedad limitada','Sociedad anonima']))
        zeros = np.zeros((data.shape[0],1))
        for i in cols:
            zeros = zeros + formas[i].values.reshape((data.shape[0],1))
            
        return pd.concat([formas[['Sociedad limitada','Sociedad anonima']],
                        pd.DataFrame(zeros.astype(int),columns=['Otras formas juridicas'])],
                        axis=1)
    
    def __dummis_variable_CNAE(self,data):
        CNAE = pd.get_dummies(data['CNAE_Number'])
        cols = list(set(data['CNAE_Number'].value_counts().index) - set(['C','G','F','M']))
        zeros = np.zeros((data.shape[0],1))
        for i in cols:
            zeros = zeros + CNAE[i].values.reshape((data.shape[0],1))
        
        return pd.concat([CNAE[['C','G','F','M']],
                        pd.DataFrame(zeros.astype(int),columns=['Otras actividades'])],
                        axis=1)

    def __prepare_to_model(self,X_train1,imp,columnas_):

    
        X_train2 = imp.transform(X_train1[columnas_])#Quitamos los valores nulos
        X_train2 = pd.DataFrame(X_train2,columns=columnas_)
        #print(X_train1.head())
        #print('##############')
        #print(X_train2.head())
        columns_part1 = list(set(list(X_train1.columns)) - set(list(columnas_)))
        #print(columns_part1)
        part1 = pd.DataFrame(X_train1[columns_part1].values,columns = columns_part1)
        #print(part1.head())
        X_train = pd.concat([part1,X_train2],axis=1)
        #X_train2.isnull().any().any()

        #print([i for i in X_train.columns])
        num = [i for i in X_train.columns].index('Ingresos de explotación EUR Últ. año disp.')
        prueba =X_train[X_train.columns[num:]]
        

        nif = pd.DataFrame(X_train[['Código NIF']].values,columns = ['Código NIF'])
        formas_juridicas = self.__dummis_variable_Formajuridica(X_train)
        CNAE = self.__dummis_variable_CNAE(X_train)
        X = pd.concat([nif,formas_juridicas,CNAE,prueba],axis=1)

        return X

    def fit(self):
        for i in ['Total Activo EUR Año - 2','Total Activo EUR Año - 3','Total Activo EUR Año - 4']:
                self.concursos[i] = self.concursos[i].astype(np.float32)
        
        self.concursos = self.concursos[((self.concursos['Total Activo EUR Año - 2'] <=126631582)|
                (self.concursos['Total Activo EUR Año - 3'] <= 126631582)|
                (self.concursos['Total Activo EUR Año - 4'] <=126631582))&
                ((self.concursos['Total Activo EUR Año - 2'] >=10777)|
                (self.concursos['Total Activo EUR Año - 3'] >=10777)|
                (self.concursos['Total Activo EUR Año - 4'] >=10777))]   
        #print(self.concursos.shape)
        cols_act = [i for i in self.activas.columns if i[-2:] == '.1']
        cols_con = [i for i in self.concursos.columns if i[-2:] == '.1']  

        self.activas = self.activas.drop(cols_act,axis=1)
        self.concursos = self.concursos.drop(cols_con,axis=1)  

        new_columns = self.__new_columns() 

        val = self.concursos.values
        self.concursos =  pd.DataFrame(val,columns=new_columns)

        self.activas = self.activas[self.concursos.columns]

        dataset = pd.concat([self.activas,self.concursos],axis=0,ignore_index=True)

        dataset = self.__delete_latest_year(dataset)

        dataset['Código primario CNAE 2009'] = dataset['Código primario CNAE 2009'].astype(np.float64)

        dataset = self.__transform_CNAE(dataset)

        dataset = dataset.drop(dataset[dataset.CNAE_Number.isnull()].index)
        dataset = self.__dataset_to_float(dataset)
        dataset = self.__drop_columns_nan(dataset)

        #dataset = dataset.drop(['Incidencias Judiciales'],axis=1)#borrar

        contador=3
        columnas=[]
        for i in range(7,len(dataset.columns)-2):
            if contador%3 == 0:
                columnas.append(dataset.columns[i][:-15])
            contador+=1
        #print(dataset.columns)
        
        self.columnas = columnas

        self.dataset = dataset
        #print(self.dataset)


    def transform(self):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        X=self.dataset.drop(['Estado'],axis=1)
        y=self.dataset['Estado']

        X_p,X_test,Y_p,Y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0) #Primera subdivision

        X_train,X_val,Y_train,Y_val = train_test_split(X_p,Y_p,test_size = 0.25, stratify = Y_p , random_state = 0)

        columnas_ = list(X_train.columns[6:-2])
        print(columnas_)
        print('#################################')
        tr = DecisionTreeRegressor(max_depth=7)
        train_imp = trainNaN(X_train,tr,self.columnas)
        train_imp.fit()

        X_train = train_imp.predict(X_train,columnas_)
        X_val =  train_imp.predict(X_val,columnas_)
        X_test = train_imp.predict(X_test,columnas_)
        print('he salido')
        Y_train = pd.DataFrame(Y_train,columns =['Estado'])
        Y_val = pd.DataFrame(Y_val,columns =['Estado'])
        Y_test = pd.DataFrame(Y_test,columns =['Estado'])

        print(columnas_)
        from sklearn.impute import SimpleImputer
        
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp = imp.fit(X_train[columnas_])

        print('he entrenado')
        datas = [X_train,X_val,X_test,Y_train,Y_val,Y_test]
        names = ['X_train','X_val','X_test','Y_train','Y_val','Y_test']

        for i in range(3):
            datas[i] = self.__prepare_to_model(datas[i],imp,columnas_)

        for i in range(len(datas)):
            datas[i].to_csv('../Datasets/iteracion_prueba/datas/'+names[i]+'.csv',index=False)

'''
mount = mount_dataset('../Datasets/iteracion_prueba/activas_','../Datasets/iteracion_prueba/datas/activas.csv',9)

mount.fit()
mount.transform()
'''
'''
mount = mount_dataset('../Datasets/iteracion_prueba/concursos-','../Datasets/iteracion_prueba/datas/concursos.csv',5)

mount.fit()
mount.transform()
'''
'''
prep = preprocesamiento('../Datasets/iteracion_prueba/datas/activas.csv','../Datasets/iteracion_prueba/datas/concursos.csv')

prep.fit()
prep.transform()
'''













