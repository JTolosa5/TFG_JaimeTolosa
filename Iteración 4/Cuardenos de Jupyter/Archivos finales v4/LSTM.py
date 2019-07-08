from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
from keras.utils import to_categorical
from keras import backend as K


def exp_decay(epoch):
    from numpy import exp
    initial_lrate = 0.01
    k = 0.3
    lrate = initial_lrate * exp(-k*epoch)
    return lrate

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        #self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.lr.append(exp_decay(len(self.losses)))

def threshold_binary_accuracy(y_true, y_pred):
    threshold = 0.7
    if K.backend() == 'tensorflow':
        #return K.mean(K.equal(y_true,K.tf.cast(y_pred >= threshold, y_true.dtype)))
        #return K.equal(y_true,K.tf.cast(y_pred >= threshold, y_true.dtype))
        y_pr = K.tf.cast(y_pred[:,0] < threshold, y_pred.dtype)
        y_tr = K.tf.cast(y_true[:,0] <= 0, y_true.dtype)
        return K.mean(K.equal(y_tr,y_pr))

def predict_threshold_binary_accuracy(y_pred):
    
    threshold = 0.7
    
    return K.tf.cast(y_pred[:,0] < threshold, y_pred.dtype)



class RNN():
    def __init__(self,callbacks_list=[],metrics=['binary_accuracy'],class_weight={0:1,1:1}):


        self.class_weight = class_weight
        self.callbacks_list = callbacks_list
        self.metrics = metrics
        self.model = None

    def fit(self,xlstm,y,epochs=30,batch_size = 50 ,lr = 0.001,hidden_layers = [5],dropout = {},validation_data = None):

        
        #target = to_categorical(y.values)

        self.model = Sequential()

        self.model.add(LSTM(hidden_layers[0],batch_input_shape=(None,3,xlstm.shape[2]),return_sequences=True))

        for i in range(1,len(hidden_layers)-1):

            self.model.add(LSTM(hidden_layers[i],return_sequences=True))
            try:
                self.model.add(Dropout(self.dropout[i]))
            except:
                pass

        self.model.add(LSTM(hidden_layers[len(hidden_layers)-1]))
        self.model.add(Dense(2,activation='softmax'))

        optimizer = keras.optimizers.Adam(lr = lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=self.metrics)
        if validation_data:

            self.model.fit(xlstm,y,epochs=epochs,batch_size=batch_size,class_weight=self.class_weight,
                        callbacks = self.callbacks_list,validation_data = validation_data,verbose=2)
        else:
            self.model.fit(xlstm,y,epochs=epochs,batch_size=batch_size,class_weight=self.class_weight,
                        validation_split = 0.1,callbacks = self.callbacks_list,verbose=2)



    def predict(self,x):
        import tensorflow as tf
        predicts_threshold = self.model.predict(x,batch_size=1)
        session = tf.Session()
        predicts_threshold = predict_threshold_binary_accuracy(predicts_threshold).eval(session = session)
        predicts = self.model.predict_classes(x)
        return predicts,predicts_threshold.reshape((x.shape[0],1))


loss = LossHistory()

lrate = keras.callbacks.LearningRateScheduler(exp_decay)
callbacks_list = [loss]
metrics_list = ['binary_accuracy',threshold_binary_accuracy]
