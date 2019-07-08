from sklearn.metrics import make_scorer

def fpr(y_act,y_pred):
    from sklearn.metrics import confusion_matrix
    
    matrix = confusion_matrix(y_act,y_pred)
    return matrix[0,0]/(matrix[0,0]+matrix[0,1])
    
def precision(y_act,y_pred):
    from sklearn.metrics import confusion_matrix
    
    matrix = confusion_matrix(y_act,y_pred)
    return matrix[0,0]/(matrix[0,0]+matrix[1,0])

def f1(y_act,y_pred):
    recall = fpr(y_act,y_pred)
    prc = precision(y_act,y_pred)
    return (2*recall*prc)/(recall+prc)

fpr_score = make_scorer(f1)

