#!/usr/bin/env python


import pandas as pd
import numpy as np
import copy

class trainNaN:
	def __init__(self,df,model,columna):

		self.df = copy.copy(df)
		self.columna = columna
		self.columnas = {col:[col+' Últ. año disp.',col+' Año - 1',col+' Año - 2'] for col in columna}
		self.model = copy.copy(model)
		self.models={}
		self.model_q={}
		
	def fit(self):
		
		#print(self.columnas)
		for col in self.columnas:
			data=self.df[self.columnas[col]]
			data=data[data.isnull().any(axis=1) == False]

			data2 = self.df[self.columnas[col]]
			data2 = data2[data2.isnull().any(axis=1)]
			
			contador=0
			models_p = []
			
			for i in self.columnas[col]:
				clon1 = copy.copy(self.model)
				clon2 = copy.copy(self.model)
				clon3 = copy.copy(self.model)
				
				y = data.iloc[:][i]
				X = data.drop([self.columnas[col][contador]],axis=1)
				
				X_p = data.iloc[:][i]
				y_0 = X.drop(X.columns[1],axis=1).iloc[:,0]
				y_1 = X.drop(X.columns[0],axis=1).iloc[:,0]
				
				model_q0 = clon1.fit(pd.DataFrame({'colum':X_p.values}),y_0)
				model_q1 = clon2.fit(pd.DataFrame({'colum':X_p.values}),y_1)
				
				models_01 = [model_q0,model_q1]
				self.model_q[i] = models_01
				
				#print(type(X.iloc[0,0]))
				#print(type(y.loc[0]))
				
				srv = clon3.fit(X,y)
				models_p.append(srv)
				contador += 1
				
			self.models[col] = models_p

			
	def predict(self,testeo,columnas):
		#dataset  =  pd.DataFrame()
		print(testeo.columns)
		print(columnas)
		test_p = testeo.drop(columnas,axis=1)
		test = copy.copy(testeo[columnas])
		#print(dataset)
		#print(self.models)
		contador = 0
		for nrow , row in test.iterrows():
			if(contador % 100 == 0 ):
				print(contador)
			for col in self.columnas:
				row_p = row[self.columnas[col]]
				#print('antes del if')
				if row_p[self.columnas[col]].isnull().sum() == 1:
					#print('hola he entrado al if')
					num = list(row_p[self.columnas[col]].isnull()).index(True)
					X = row_p[list(set(self.columnas[col])-set([self.columnas[col][num]]))]
					model_=self.models[col][num]
					#print(model_)
					row[self.columnas[col][num]]=model_.predict([X.values])[0]
					#print(row)
					#print(test)
					
				elif row_p[self.columnas[col]].isnull().sum() == 2:
					#print('hola he entrado al elif')
					null = [i for i in range(3) if row_p[self.columnas[col]].isnull()[i] == True]
					num = list(row_p[self.columnas[col]].isnull()).index(False)
					
					X = [row_p[self.columnas[col][num]]]
					#print(X)
					
					model_0 = self.model_q[self.columnas[col][num]][0]
					#print(self.model_q[self.columnas[col][num]][0].feature_importances_)
					row[self.columnas[col][null[0]]] = model_0.predict([X])[0]

				
					model_1 = self.model_q[self.columnas[col][num]][1]
					
					row[self.columnas[col][null[1]]] = model_1.predict([X])[0]
					#print(row)
					#print(test)


					
			contador+=1        
			#print(dataset)
			#data = pd.DataFrame(row)
			#dataset =  pd.concat([dataset,data],axis=1)

				
		test = pd.concat([test_p,test],axis=1)        
		return test
