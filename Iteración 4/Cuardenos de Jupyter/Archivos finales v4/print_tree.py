import graphviz 
from sklearn import tree
from IPython.display import Image
from pydotplus import graph_from_dot_data  
def imprimeArbol(model, X_train, y_train, width):
    fnames = X_train.columns.values
    lnames = [str(x) for x in y_train.unique()]
    dot_data = tree.export_graphviz(model, out_file=None, 
                             feature_names=fnames,  
                             class_names=lnames,  
                             filled=True, rounded=True,  
                             special_characters=True,
                             )  
    graph = graphviz.Source(dot_data) 
    graph2 = graph_from_dot_data(dot_data)
    graph2.write_png('tree.png')
    return graph
