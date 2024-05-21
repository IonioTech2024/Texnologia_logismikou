import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to load the data from a file
def load_data(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        return None

# Function to do some cool 2D plotting
def plot_2d(data, labels, algorithm='PCA'):
    if algorithm == 'PCA':
        model = PCA(n_components=2)
    elif algorithm == 't-SNE':
        model = TSNE(n_components=2)
    else:
        st.error("Unsupported algorithm")
        return None

    transformed = model.fit_transform(data)
    df = pd.DataFrame(transformed, columns=['Component 1', 'Component 2'])
    df['Label'] = labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Label', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

# Main app title
st.title('Data Visualization and Machine Learning with Streamlit')

# Setting up the tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Data Visualization", "Machine Learning", "Your Name"])

with tab1:
    st.header("Home")
    st.write("Welcome to the Data Visualization and Machine Learning App!")

# File uploader on the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # Splitting features and labels
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        with tab1:
            st.write("Here's a preview of your data:")
            st.dataframe(data)  # Display all the data

            st.write("General Distribution Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=y, kde=True, ax=ax)
            st.pyplot(fig)

        with tab2:
            # 2D Visualization Tab
            st.sidebar.header("2D Visualization")
            vis_algo = st.sidebar.selectbox("Pick your algorithm", ['PCA', 't-SNE'])
            
            st.subheader("2D Visualization")
            st.write("Choose the algorithm and see your data in 2D")
            plot_2d(X, y, vis_algo)
            
            # Exploratory Data Analysis (EDA)
            st.sidebar.header("Exploratory Data Analysis")
            st.subheader("Exploratory Data Analysis")
            st.write("Select a chart to explore your data")
            chart_type = st.sidebar.selectbox("Pick a chart type", ['Correlation Heatmap', 'Pairplot', 'Distribution Plot'])

            if chart_type == 'Correlation Heatmap':
                st.write("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            elif chart_type == 'Pairplot':
                st.write("Pairplot")
                sns.pairplot(data, hue=data.columns[-1])
                st.pyplot()

            elif chart_type == 'Distribution Plot':
                st.write("Distribution Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=y, kde=True, ax=ax)
                st.pyplot(fig)

        with tab3:
            # Machine Learning Tabs
            st.sidebar.header("Machine Learning")
            ml_task = st.sidebar.selectbox("Pick your ML task", ['Classification', 'Clustering'])

            if ml_task == 'Classification':
                st.subheader("Classification")
                test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
                random_state = st.sidebar.slider("Random State", 0, 100, 42)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                model = RandomForestClassifier(random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.write(f"Classification Accuracy: {accuracy:.2f}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
                st.pyplot()

            elif ml_task == 'Clustering':
                st.subheader("Clustering")
                n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X)

                st.write("Cluster Visualization")
                plot_2d(X, clusters, 'PCA')

with tab4:
    # Extra tab to just show your name
    st.header("Your Name")
    st.write("Stergios Moutzikos")  # Replace "John Doe" with your actual name
