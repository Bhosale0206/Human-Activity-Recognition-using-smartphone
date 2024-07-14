import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Suppress warnings globally
import warnings
warnings.filterwarnings("ignore")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.xticks(rotation=90)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

# Function to get best random search results
def get_best_randomsearch_results(model):
    st.subheader("Best estimator:")
    st.write(model.best_estimator_)
    st.subheader("Best set of parameters:")
    st.write(model.best_params_)
    # Uncomment below if you want to display best score
    st.subheader("Best score:")
    st.write(model.best_score_)

# Load data with error handling
@st.cache  # Cache the data so that it doesn't reload on every interaction
def load_data(train_path, test_path):
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        # Specify the path to your image folder
        image_folder = 'C:\\Users\\ADMIN\\Desktop\\HARUS\\human+activity+recognition+using+smartphones\\UCI HAR Dataset\\UCI HAR Dataset\\images'
        
        # Add image paths to train and test data
        train['Image_Path'] = train['Activity'].apply(lambda x: os.path.join(image_folder, f'{x}.GIF'))  # Adjust file extension if needed
        test['Image_Path'] = test['Activity'].apply(lambda x: os.path.join(image_folder, f'{x}.GIF'))  # Adjust file extension if needed
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None
    return train, test

# Main function to run the Streamlit app
def main():
    st.title("Human Activity Recognition - Machine Learning App")
    
    # Load data
    train, test = load_data('train.csv', 'test.csv')
    if train is None or test is None:
        return
    
    # Sidebar options
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "EDA", "Model Building", "About", "Activity Images"])
    
    if page == "Homepage":
        st.subheader("Training Data Overview")
        st.write(train.head())
        st.subheader("Test Data Overview")
        st.write(test.head())
        
    elif page == "EDA":
        st.subheader("Exploratory Data Analysis")
        st.write("Number of duplicates in train : ", sum(train.duplicated()))
        st.write("Number of duplicates in test : ", sum(test.duplicated()))
        st.write("Total number of missing values in train : ", train.isna().values.sum())
        st.write("Total number of missing values in test : ", test.isna().values.sum())
        
        # Visualization
        st.subheader("Distribution of Activities in Train Dataset")
        activity_counts_train = train['Activity'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(activity_counts_train, labels=activity_counts_train.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
        plt.title('Distribution of Activities in Train Dataset')
        plt.axis('equal')
        st.pyplot()
        
        st.subheader("Distribution of Subjects in Train Dataset")
        subject_counts_train = train['subject'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(subject_counts_train, labels=subject_counts_train.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
        plt.title('Distribution of Subjects in Train Dataset')
        plt.axis('equal')
        st.pyplot()
        
    elif page == "Model Building":
        st.subheader("Machine Learning Models")
        
        # Prepare data
        X_train = train.drop(['subject', 'Activity', 'Image_Path'], axis=1)  # Assuming you don't need Image_Path for modeling
        y_train = train['Activity']
        X_test = test.drop(['subject', 'Activity', 'Image_Path'], axis=1)
        y_test = test['Activity']
        
        # Choose model
        model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Kernel SVM", "Decision Tree", "Random Forest"])
        
        if model_choice == "Logistic Regression":
            st.subheader("Logistic Regression Model")
            
            # Hyperparameter tuning
            parameters = {'max_iter': [100, 200, 500]}
            lr_classifier = LogisticRegression()
            lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters, cv=5, random_state=42)
            lr_classifier_rs.fit(X_train, y_train)
            
            # Predictions and accuracy
            y_pred_lr = lr_classifier_rs.predict(X_test)
            lr_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_lr)
            st.write("Accuracy using Logistic Regression:", lr_accuracy)
            
            # Confusion Matrix
            cm_lr = confusion_matrix(y_test, y_pred_lr)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm_lr, np.unique(y_pred_lr))
            st.pyplot()
            
            # Best parameters
            st.subheader("Best parameters found by Randomized Search:")
            get_best_randomsearch_results(lr_classifier_rs)
        
        elif model_choice == "Kernel SVM":
            st.subheader("Kernel SVM Model")
            
            # Hyperparameter tuning
            parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [100, 50]}
            svm_classifier = SVC()
            svm_classifier_rs = RandomizedSearchCV(svm_classifier, param_distributions=parameters, cv=5, random_state=42)
            svm_classifier_rs.fit(X_train, y_train)
            
            # Predictions and accuracy
            y_pred_svm = svm_classifier_rs.predict(X_test)
            svm_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_svm)
            st.write("Accuracy using Kernel SVM:", svm_accuracy)
            
            # Confusion Matrix
            cm_svm = confusion_matrix(y_test, y_pred_svm)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm_svm, np.unique(y_pred_svm))
            st.pyplot()
            
            # Best parameters
            st.subheader("Best parameters found by Randomized Search:")
            get_best_randomsearch_results(svm_classifier_rs)
        
        elif model_choice == "Decision Tree":
            st.subheader("Decision Tree Model")
            
            # Hyperparameter tuning
            parameters = {'max_depth': np.arange(2, 10, 2)}
            dt_classifier = DecisionTreeClassifier()
            dt_classifier_rs = RandomizedSearchCV(dt_classifier, param_distributions=parameters, cv=5, random_state=42)
            dt_classifier_rs.fit(X_train, y_train)
            
            # Predictions and accuracy
            y_pred_dt = dt_classifier_rs.predict(X_test)
            dt_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_dt)
            st.write("Accuracy using Decision Tree:", dt_accuracy)
            
            # Confusion Matrix
            cm_dt = confusion_matrix(y_test, y_pred_dt)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm_dt, np.unique(y_pred_dt))
            st.pyplot()
            
            # Best parameters
            st.subheader("Best parameters found by Randomized Search:")
            get_best_randomsearch_results(dt_classifier_rs)
        
        elif model_choice == "Random Forest":
            st.subheader("Random Forest Model")
            
            # Hyperparameter tuning
            parameters = {'n_estimators': np.arange(20, 101, 10), 'max_depth': np.arange(2, 17, 2)}
            rf_classifier = RandomForestClassifier()
            rf_classifier_rs = RandomizedSearchCV(rf_classifier, param_distributions=parameters, cv=5, random_state=42)
            rf_classifier_rs.fit(X_train, y_train)
            
            # Predictions and accuracy
            y_pred_rf = rf_classifier_rs.predict(X_test)
            rf_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_rf)
            st.write("Accuracy using Random Forest:", rf_accuracy)
            
            # Confusion Matrix
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm_rf, np.unique(y_pred_rf))
            st.pyplot()
            
            # Best parameters
            st.subheader("Best parameters found by Randomized Search:")
            get_best_randomsearch_results(rf_classifier_rs)
    
    elif page == "Activity Images":
        st.subheader("Activity Images")
        
        activity = st.selectbox("Select an activity", train['Activity'].unique())
        image_path = train.loc[train['Activity'] == activity, 'Image_Path'].iloc[0]
        
        # Display image or gif
        if image_path.lower().endswith('.gif'):
            with open(image_path, 'rb') as f:
                contents = f.read()
                st.image(contents, caption=f"Activity: {activity}", use_column_width=True)
        else:
            try:
                image = plt.imread(image_path)
                st.image(image, caption=f"Activity: {activity}", use_column_width=True)
            except FileNotFoundError:
                st.error(f"Error opening '{image_path}'")
    
    elif page == "About":
        st.subheader("About This App")
        st.markdown("""
            This web app demonstrates various machine learning models for Human Activity Recognition using the UCI HAR Dataset.
            - **EDA**: Exploratory Data Analysis to understand the dataset.
            - **Model Building**: Building and comparing different ML models.
            - **Deployment**: Streamlit for interactive deployment.
            """)
    
# Run the main function
if __name__ == "__main__":
    main()
