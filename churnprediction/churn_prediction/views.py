import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from .models import Customer
from .forms import UploadFileForm
from data_processing.preprocessing import clean_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from .models import Customer
import os


def upload_customer_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle file upload
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            # Read the file into a DataFrame
            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')

                # Process and save each row to the Customer model
                for index, row in df.iterrows():
                    customer_id = row['customerID']

                    # Convert values, handling empty or whitespace strings
                    def safe_float(value):
                        if isinstance(value, str):
                            value = value.strip()  # Remove leading/trailing whitespace
                        return float(value) if value else None

                    # Use the safe_float function for conversions
                    monthly_charges = safe_float(row['MonthlyCharges'])
                    total_charges = safe_float(row['TotalCharges'])

                    # Check if customer already exists
                    customer, created = Customer.objects.get_or_create(
                        customerID=customer_id,
                        defaults={
                            'gender': row['gender'],
                            'senior_citizen': bool(row['SeniorCitizen']),
                            'partner': bool(row['Partner']),
                            'dependents': bool(row['Dependents']),
                            'tenure': int(row['tenure']),
                            'phone_service': bool(row['PhoneService']),
                            'multiple_lines': row['MultipleLines'],
                            'internet_service': row['InternetService'],
                            'online_security': row['OnlineSecurity'],
                            'online_backup': row['OnlineBackup'],
                            'device_protection': row['DeviceProtection'],
                            'tech_support': row['TechSupport'],
                            'streaming_tv': row['StreamingTV'],
                            'streaming_movies': row['StreamingMovies'],
                            'contract': row['Contract'],
                            'paperless_billing': bool(row['PaperlessBilling']),
                            'payment_method': row['PaymentMethod'],
                            'monthly_charges': monthly_charges,
                            'total_charges': total_charges,
                            'churn': bool(row['Churn']),
                        }
                    )
                    
                    # If created is False, the customer already exists, and you can handle updates if needed.
                    if not created:
                        # Optionally, update fields here if necessary
                        customer.gender = row['gender']
                        customer.senior_citizen = bool(row['SeniorCitizen'])
                        customer.partner = bool(row['Partner'])
                        customer.dependents = bool(row['Dependents'])
                        customer.tenure = int(row['tenure'])
                        customer.phone_service = bool(row['PhoneService'])
                        customer.multiple_lines = row['MultipleLines']
                        customer.internet_service = row['InternetService']
                        customer.online_security = row['OnlineSecurity']
                        customer.online_backup = row['OnlineBackup']
                        customer.device_protection = row['DeviceProtection']
                        customer.tech_support = row['TechSupport']
                        customer.streaming_tv = row['StreamingTV']
                        customer.streaming_movies = row['StreamingMovies']
                        customer.contract = row['Contract']
                        customer.paperless_billing = bool(row['PaperlessBilling'])
                        customer.payment_method = row['PaymentMethod']
                        customer.monthly_charges = monthly_charges
                        customer.total_charges = total_charges
                        customer.churn = bool(row['Churn'])
                        customer.save()  # Save updates

                return redirect('customer_list')  # Redirect after successful upload
    else:
        form = UploadFileForm()

    return render(request, 'upload_customer_data.html', {'form': form})

def customer_list(request):
    customers = Customer.objects.all()
    return render(request, 'customer_list.html', {'customers': customers})
def prediction_results(request):
    # Load customer data from the database
    customers = Customer.objects.all()
    customer_df = pd.DataFrame(list(customers.values()))

    # Ensure 'customerID' is excluded from numeric conversions
    customer_id_col = customer_df['customerID']  # Store 'customerID' separately

    # Convert relevant columns to numeric, excluding 'customerID'
    customer_df['monthly_charges'] = pd.to_numeric(customer_df['monthly_charges'], errors='coerce')
    customer_df['total_charges'] = pd.to_numeric(customer_df['total_charges'], errors='coerce')

    # Handle NaNs after conversion
    customer_df.fillna(0, inplace=True)

    # Data cleaning and preprocessing (excluding 'customerID')
    customer_df = clean_data(customer_df.drop(columns=['customerID']))

    # Reattach 'customerID' after preprocessing
    customer_df['customerID'] = customer_id_col

    # Ensure that 'customerID' is not included in the feature set for model training
    X = customer_df.drop(columns=['churn', 'customerID'])  # Drop 'customerID' and 'churn' from features
    y = customer_df['churn']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection and training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save the trained model
    model_path = 'models/churn_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Filter for a specific customer based on Customer ID if provided
    customer_id = request.GET.get('customer_id', None)
    if customer_id:
        customer_df = customer_df[customer_df['customerID'].astype(str).str.contains(customer_id)]

    # If there are no records for the searched customer
    if customer_df.empty:
        return render(request, 'error.html', {'message': f'No customer found with ID: {customer_id}'})

    # Make predictions on the filtered data
    X_test_filtered = customer_df.drop(columns=['churn', 'customerID'], errors='ignore')

    # Combine predictions with customer IDs
    predictions = dict(zip(customer_df['customerID'], model.predict(X_test_filtered)))

    # Render the results
    return render(request, 'prediction_results.html', {
        'report': {
            'predictions': predictions,
            'accuracy': model.score(X_test_filtered, customer_df['churn']) if 'churn' in customer_df.columns else 'N/A',
            'weighted_avg': {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
            }
        }
    })
