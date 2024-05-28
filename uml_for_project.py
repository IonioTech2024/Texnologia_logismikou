import pygraphviz as pgv
from IPython.display import Image

# Define the UML diagram in DOT language
uml_code = """
digraph G {
    rankdir=LR;
    node [shape=record, style=filled, fillcolor=lightblue];
    Streamlit [label="{Streamlit|uploaded_file: FileUploader|data: DataFrame|X: DataFrame|y: DataFrame}|load_data(file: File) : DataFrame|plot_2d(data: DataFrame, labels: DataFrame, algorithm: String) : void"];
    FileUploader [label="{FileUploader|name: String|type: List<String>|read() : BytesIO}"];
    DataFrame [label="{DataFrame|#_data: Object|#_internal_names: List<String>|#_internal_names_set: Set<String>|#_metadata: List<String>|#_selected_obj: Object|#_selection: Object|#_selection_list: List<String>|#_set_is_copy: List<String>|iloc() : DataFrame|corr() : DataFrame|select_dtypes(include: List<String>) : DataFrame}"];
    PCA [label="{PCA|fit_transform(data: DataFrame) : DataFrame}"];
    TSNE [label="{TSNE|fit_transform(data: DataFrame) : DataFrame}"];
    RandomForestClassifier [label="{RandomForestClassifier|fit(X_train: DataFrame, y_train: DataFrame) : void|predict(X_test: DataFrame) : DataFrame}"];
    SVC [label="{SVC|fit(X_train: DataFrame, y_train: DataFrame) : void|predict(X_test: DataFrame) : DataFrame}"];
    LogisticRegression [label="{LogisticRegression|fit(X_train: DataFrame, y_train: DataFrame) : void|predict(X_test: DataFrame) : DataFrame}"];
    KMeans [label="{KMeans|fit_predict(X: DataFrame) : DataFrame|#inertia_: float|cluster_centers_ : DataFrame}"];
    Streamlit -> FileUploader [arrowhead=vee];
    Streamlit -> DataFrame [arrowhead=vee];
    Streamlit -> PCA [arrowhead=vee];
    Streamlit -> TSNE [arrowhead=vee];
    Streamlit -> RandomForestClassifier [arrowhead=vee];
    Streamlit -> SVC [arrowhead=vee];
    Streamlit -> LogisticRegression [arrowhead=vee];
    Streamlit -> KMeans [arrowhead=vee];
    FileUploader -> BytesIO [style=dotted, arrowhead=vee];
    DataFrame -> Object [style=dotted];
    PCA -> DataFrame [style=dotted];
    TSNE -> DataFrame [style=dotted];
    RandomForestClassifier -> DataFrame [style=dotted];
    SVC -> DataFrame [style=dotted];
    LogisticRegression -> DataFrame [style=dotted];
    KMeans -> DataFrame [style=dotted];
}
"""

# Create a graph from the UML code
graph = pgv.AGraph(string=uml_code)

# Render the graph to an image file
graph.draw('uml_diagram.png', prog='dot', format='png')

# Display the image in Jupyter Notebook
Image(filename='uml_diagram.png')
