from sklearn.metrics import f1_score

def calculate_f_measure(y_true,y_pred):
    f1_measure = f1_score(y_true,y_pred,average='weighted')
    return f1_measure