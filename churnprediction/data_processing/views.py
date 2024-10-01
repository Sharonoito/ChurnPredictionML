import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from data_processing.preprocessing import clean_data, split_data, train_model, evaluate_model

def upload_and_predict(request):
    if request.method == 'POST' and request.FILES['file']:
        # Handle file upload
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)
        
        # Load the file into a DataFrame
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Unsupported file type
            return render(request, 'upload_customer_data.html', {'error': "Unsupported file type. Please upload a .csv or .xlsx file."})

        # Step 1: Clean and preprocess data
        cleaned_df = clean_data(df)
        
        # Step 2: Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(cleaned_df)
        
        # Step 3: Train the model (using Logistic Regression for now)
        model = train_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Make predictions
        y_pred = model.predict(X_test)
        
        # Step 5: Evaluate the model
        evaluate_model(y_test, y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        # Return predictions and accuracy to template
        return render(request, 'prediction_results.html', {
            'accuracy': accuracy,
            'predictions': y_pred
        })
    
    return render(request, 'upload_customer_data.html')

