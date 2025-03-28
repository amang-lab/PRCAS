import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,roc_curve, roc_auc_score, confusion_matrix,auc
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
import time

"""
This is the file that train the data from excel file
"""

def log(X_train,X_test,y_train,y_test):

    # Define the hyperparameters to tune
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type of regularization
        'solver': ['liblinear', 'saga']  # Solvers compatible with penalties
    }
    # Define the logistic regression model
    log_reg = LogisticRegression(max_iter=5000, random_state=0)
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # Metric to optimize (e.g., accuracy)
        verbose=1,  # Show progress
        n_jobs=-1  # Use all available processors
    )

    # Fit the model to find the best parameters
    grid_search.fit(X_train, y_train.values.ravel())

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)

    # Evaluate the performance
    print("\nClassification Report:\n", classification_report(y, y_pred))
def FindP(X_train, X_test,y_train,y_test):
    # Define logistic regression model
    log_reg = LogisticRegression(max_iter=5000, random_state=0, multi_class='multinomial')

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['newton-cg', 'sag', 'lbfgs']  # Solvers compatible with multiclass problems
    }

    # Initialize GridSearchCV with scoring='roc_auc_ovr' for multiclass AUC
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        scoring='roc_auc_ovr',  # Use One-vs-Rest ROC AUC for multiclass problems
        cv=5,  # 5-fold cross-validation
        verbose=1,  # Display progress
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train.values.ravel())

    # Print the best parameters and corresponding roc_auc score
    print("Best Parameters:", grid_search.best_params_)
    print("Best roc_auc (One-vs-Rest) Score:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Optionally, print a classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
def lgbmc(X_train, X_test,y_train,y_test):
    lgbm = lgb.LGBMClassifier(random_state=0,n_jobs=-1)

    # Define the hyperparameter grid
    param_grid = {
        'num_leaves': [15, 31, 63],  # Number of leaves in the tree
        'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
        'n_estimators': [100, 200, 500],  # Number of boosting iterations
        'max_depth': [3, 5, 10],  # Maximum tree depth
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring='accuracy',  # Change to other metrics like 'roc_auc' if needed
        cv=5,  # 5-fold cross-validation
        verbose=1,  # Print progress during grid search
        n_jobs=-1  # Use all available CPU cores for parallel computation
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train.values.ravel())

    # Print the best parameters and corresponding score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def rf(X_train, X_test,y_train,y_test):
    # Define the Random Forest Classifier
    rf = RandomForestClassifier(random_state=0)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],  # Number of trees in the forest
        'max_depth': [5, 10, 20, None],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
        'bootstrap': [True, False]  # Whether to bootstrap samples when building trees
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='accuracy',  # Optimize for accuracy. Replace with 'roc_auc' for imbalanced datasets
        cv=5,  # 5-fold cross-validation
        verbose=1,  # Print progress during search
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train.values.ravel())

    # Print the best hyperparameters and their cross-validation score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Print the classification report for the test set
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def svmm(X_train2, X_test,y_train2,y_test):
    # Define the SVM model
    svm = SVC(probability=True, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient (used for non-linear kernels like rbf)
        'degree': [2, 3, 4]  # Degree of the polynomial kernel (only used for 'poly')
    }

    # Set up GridSearchCV
    grid_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_grid,
        scoring='accuracy',
        n_iter=10,  # Number of parameter combinations to try
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train.values.ravel())

    # Print the best hyperparameters and corresponding score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    # Print the classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def AUC(X_train2, X_test,y_train2,y_test):
    undersampler = SMOTE(random_state=0)
    X_train, y_train = undersampler.fit_resample(X_train2,y_train2)

    log_reg = LogisticRegression(max_iter=5000,random_state=0,solver='saga',C=1,penalty='l1')
    dec_tree = lgb.LGBMClassifier(random_state=0,n_jobs=-1,learning_rate=0.01, max_depth=3,n_estimators=100,num_leaves=15)
    rand_forest = RandomForestClassifier(random_state=0,bootstrap=True,max_depth=5,min_samples_leaf=1,min_samples_split=10,n_estimators=200)
    svm_model = SVC(probability=True,random_state=0,gamma='scale',degree=4,C=10)

    # Train models
    start = time.time()
    log_reg.fit(X_train, y_train.values.ravel())
    stop = time.time()
    print(f'log time {stop-start}s')
    start = time.time()
    dec_tree.fit(X_train, y_train.values.ravel())
    stop = time.time();
    print(f'lgbm time {stop-start}s')
    start = time.time()
    rand_forest.fit(X_train, y_train.values.ravel())
    stop = time.time()
    print(f'rand forest time {stop-start}s')
    start = time.time()
    svm_model.fit(X_train, y_train.values.ravel())
    stop = time.time()
    print(f'svm time {stop-start}s')

    #predict
    y_pred = log_reg.predict(X_test)
    joblib.dump(log_reg, 'LogisticRegression.pkl')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('log_reg Report:')
    print(classification_report(y_test, y_pred, digits=4))
    cm_lr = confusion_matrix(y_test,y_pred)

    y_pred = dec_tree.predict(X_test)
    joblib.dump(dec_tree, 'Gradient Boosting.pkl')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Gradient Boosting Report:')
    print(classification_report(y_test, y_pred, digits=4))
    cm_dt = confusion_matrix(y_test,y_pred)

    y_pred = rand_forest.predict(X_test)
    joblib.dump(rand_forest, 'randomforestclassifier.pkl')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:4f}')
    print('rand_forest Report:')
    print(classification_report(y_test, y_pred, digits=4))
    cm_rf = confusion_matrix(y_test,y_pred)

    y_pred = svm_model.predict(X_test)
    joblib.dump(svm_model, 'SVM.pkl')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('svm Report:')
    print(classification_report(y_test, y_pred, digits=4,zero_division=0))
    cm_svm = confusion_matrix(y_test,y_pred)


    # Predict probabilities
    y_probs_log_reg = log_reg.predict_proba(X_test)[:, 1]
    y_probs_dec_tree = dec_tree.predict_proba(X_test)[:, 1]
    y_probs_rand_forest = rand_forest.predict_proba(X_test)[:, 1]
    y_probs_svm = svm_model.predict_proba(X_test)[:,1]

    test_df = pd.DataFrame({'True':y_test.values.ravel(),
                            'Logistic Regression':y_probs_log_reg,
                           'Gradient Boosting':y_probs_dec_tree,
                            'Random Forest':y_probs_rand_forest,
                            'Support Vector Machines': y_probs_svm
                            })



    # Assuming 'True' column contains multiclass labels
    classes = test_df['True'].unique()
    y_true = label_binarize(test_df['True'], classes=classes)  # Binarize the true labels

    models = ['Logistic Regression', 'Gradient Boosting', 'Random Forest', 'Support Vector Machines']
    # Iterate over models

    # Plot ROC curves
    # Create a figure for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots
    axes = axes.ravel()
    for idx,model in enumerate(models):
        # Ensure predictions are properly structured for the current model
        if model in test_df.columns:  # Check that the model exists as a column
            y_score = np.array(test_df[model])  # Extract predictions for the current model

            ax = axes[idx]
            optimal_thresholds = []
            # Compute ROC and AUC for each class
            for i, class_label in enumerate(classes):
                fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score)
                roc_auc = auc(fpr, tpr)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                optimal_thresholds.append(optimal_threshold)
                print(f"{model} Class {class_label}  Optimal Threshold: {optimal_threshold}")
                ax.plot(fpr, tpr, label=f'{model} Class {class_label} AUC = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'\n{model} ROC Curve Comparison')
            ax.legend(loc='lower right')
    # Add labels and legend
    plt.tight_layout()
    plt.show()





    # confusion matrix
    class_names = np.array(['safe', 'attention', 'dangerous'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], xticklabels=class_names,
                yticklabels=class_names)
    axes[0, 0].set_title('Logistic Regression')

    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], xticklabels=class_names,
                yticklabels=class_names)
    axes[0, 1].set_title('Gradient Boosting')

    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], xticklabels=class_names,
                yticklabels=class_names)
    axes[1, 0].set_title('Random Forest')

    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], xticklabels=class_names,
                yticklabels=class_names)
    axes[1, 1].set_title('Support Vector Machine')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Load the data
    data = pd.read_excel('tracking.xlsx')

    # Select features and target
    X = data[['degree','zone','L/R','dist']]
    y = data[['highest risk']]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    #log(X_train,X_test,y_train,y_test)
    AUC(X_train,X_test,y_train,y_test)
    #lgbmc(X_train,X_test,y_train,y_test)
    #rf(X_train,X_test,y_train,y_test)
    #svmm(X_train,X_test,y_train,y_test)



