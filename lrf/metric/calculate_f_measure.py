from sklearn.metrics import f1_score

def calculate_f_measure(y_true,y_pred):
    import numpy as np
    f1_measure = f1_score(np.asarray(y_true),np.asarray(y_pred),y_pred,average='weighted')
    return f1_measure