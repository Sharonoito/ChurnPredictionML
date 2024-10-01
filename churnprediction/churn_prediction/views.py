import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from .models import Customer
from .forms import UploadFileForm
from data_processing.preprocessing import clean_data, split_data, train_model, evaluate_model  # Import the ML functions
import joblib
import os
from django.conf import settings

def upload_customer_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Step 1: Handle file upload
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            # Step 2: Check file type and read into DataFrame
            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                form.add_error(None, "Unsupported file type. Please upload a .csv or .xlsx file.")
                return render(request, 'upload_customer_data.html', {'form': form})

            # Step 3: Clean and preprocess the data
            df = clean_data(df)  # Clean and preprocess the uploaded data

            # Step 4: Save each customer record to the database
            for index, row in df.iterrows():
                # Convert total_charges to float and handle missing data
                total_charges_str = str(row['TotalCharges']).strip()  # Convert to string and strip spaces
                if total_charges_str and total_charges_str.replace('.', '', 1).isdigit():
                    total_charges = float(total_charges_str)
                else:
                    total_charges = None

                # Check if customer already exists
                if not Customer.objects.filter(customerID=row['customerID']).exists():
                    Customer.objects.create(
                        customerID=row['customerID'],
                        gender=row['gender'],
                        senior_citizen=bool(row['SeniorCitizen']),
                        partner=bool(row['Partner']),
                        dependents=bool(row['Dependents']),
                        tenure=int(row['tenure']),
                        phone_service=bool(row['PhoneService']),
                        multiple_lines=row['MultipleLines'],
                        internet_service=row['InternetService'],
                        online_security=row['OnlineSecurity'],
                        online_backup=row['OnlineBackup'],
                        device_protection=row['DeviceProtection'],
                        tech_support=row['TechSupport'],
                        streaming_tv=row['StreamingTV'],
                        streaming_movies=row['StreamingMovies'],
                        contract=row['Contract'],
                        paperless_billing=bool(row['PaperlessBilling']),
                        payment_method=row['PaymentMethod'],
                        monthly_charges=float(row['MonthlyCharges']),
                        total_charges=total_charges,  # Use the parsed total_charges value
                        churn=bool(row['Churn'])
                    )

            # Step 5: Split the data for model training
            X_train, X_test, y_train, y_test = split_data(df)

            # Step 6: Train the model (you can choose between logistic regression or random forest)
            model = train_model(X_train, y_train, X_test, y_test)

            # Step 7: Evaluate the model
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred)

            model_path = os.path.join(settings.BASE_DIR, 'churn_model.pkl')  # Use BASE_DIR for full path
            joblib.dump(model, model_path)

              # Step 8: Prepare predictions for display
            predictions = list(zip(X_test.index, y_pred))  # Combine the indices with predictions

            # Step 9: Render the results page with the predictions
            return render(request, 'prediction_results.html', {'predictions': predictions})

            # return redirect('customer_list')
    else:
        form = UploadFileForm()

    return render(request, 'upload_customer_data.html', {'form': form})


def customer_list(request):
    # This view lists all the customers from the database
    customers = Customer.objects.all()
    return render(request, 'customer_list.html', {'customers': customers})


def dashboard_view(request):
    # View for rendering the dashboard
    customers = Customer.objects.all()
    
    # Apply filters if provided
    customerID = request.GET.get('customerID')
    churnRisk = request.GET.get('churnRisk')

    if customerID:
        customers = customers.filter(customerID__icontains=customerID)
    if churnRisk:
        customers = customers.filter(churn_predictions__churn_risk=churnRisk)

    # Prepare data for charts (you can adjust this part based on your actual churn risk field)
    churn_risk_data = {
        'high': customers.filter(churn_predictions__churn_risk='high').count(),
        'medium': customers.filter(churn_predictions__churn_risk='medium').count(),
        'low': customers.filter(churn_predictions__churn_risk='low').count()
    }

    context = {
        'customers': customers,
        'churn_risk_data': churn_risk_data
    }
    return render(request, 'prediction_dashboard.html', context)

def prediction_results(request):
    # Fetch all customers from the database
    customers = Customer.objects.all()

    # Convert the customer data into a DataFrame
    customer_df = pd.DataFrame(list(customers.values()))

    # Check the original columns in the DataFrame
    print("Original Columns in DataFrame:", customer_df.columns.tolist())

    # Normalize column names to lowercase
    customer_df.columns = customer_df.columns.str.lower()  # Convert to lowercase
    print("Normalized Columns in DataFrame:", customer_df.columns.tolist())  # Check columns again

    # Preprocess the data
    customer_df = clean_data(customer_df)  # Ensure data is cleaned and preprocessed
    print("Columns after cleaning:", customer_df.columns.tolist())  # Check columns after cleaning

    # Ensure 'churn' exists before splitting
    if 'churn' in customer_df.columns:  # Check for lowercase 'churn'
        # Proceed with the split_data function if 'churn' exists
        X_train, X_test, y_train, y_test = split_data(customer_df)
    else:
        # Handle case when 'churn' does not exist
        print("'churn' column not found in DataFrame.")
        # If you don't have actual 'churn' labels, use all data for predictions
        X_test = customer_df

    # Load the pre-trained model
    model_path = os.path.join(settings.BASE_DIR, 'churn_model.pkl')  # Use BASE_DIR for full path
    
    if not os.path.exists(model_path):
        print(f"Model file does not exist at: {model_path}")
        return render(request, 'error.html', {'message': 'Model file not found. Please train the model first.'})

    model = joblib.load(model_path)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Combine predictions with customer IDs (or relevant information)
    predictions = list(zip(customer_df['customerid'], y_pred))

    # Render the prediction results in the template
    return render(request, 'prediction_results.html', {'predictions': predictions})

# def prediction_dashboard(request):
#     customers = Customer.objects.all()
    
#     customerID = request.GET.get('customerID')
#     churnRisk = request.GET.get('churnRisk')

#     if customerID:
#         customers = customers.filter(customerID__icontains=customerID)
#     if churnRisk:
#         customers = customers.filter(churnRisk=churnRisk)

#     churn_risk_data = {
#         'high': customers.filter(churn_risk='high').count(),
#         'medium': customers.filter(churn_risk='medium').count(),
#         'low': customers.filter(churn_risk='low').count()
#     }
    
#     context = {
#         'customers': customers,
#         'churn_risk_data': churn_risk_data
#     }
#     return render(request, 'prediction_dashboard.html', context)