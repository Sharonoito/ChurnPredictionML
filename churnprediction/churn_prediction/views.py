# churn_prediction/views.py
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from .models import Customer
from .forms import UploadFileForm

def upload_customer_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle file upload
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            # Check file type (either .xlsx or .csv)
            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # Unsupported file type
                form.add_error(None, "Unsupported file type. Please upload a .csv or .xlsx file.")
                return render(request, 'upload_customer_data.html', {'form': form})

            # Process and save each row to the Customer model
            for index, row in df.iterrows():
                # Convert total_charges to float, handle edge cases
                total_charges_str = str(row['TotalCharges']).strip()  # Convert to string and strip spaces
                if total_charges_str and total_charges_str.replace('.', '', 1).isdigit():
                    total_charges = float(total_charges_str)
                else:
                    total_charges = None

                # Check if customer already exists before creating
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

            return redirect('customer_list')  # Redirect after successful upload
    else:
        form = UploadFileForm()

    return render(request, 'upload_customer_data.html', {'form': form})


def customer_list(request):
    customers = Customer.objects.all()
    return render(request, 'customer_list.html', {'customers': customers})



# # churn_prediction/views.py
# import pandas as pd
# from django.core.files.storage import FileSystemStorage
# from django.shortcuts import render, redirect
# from .models import Customer
# from .forms import UploadFileForm


# def upload_customer_data(request):
#     if request.method == 'POST':
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Handle file upload
#             file = request.FILES['file']
#             fs = FileSystemStorage()
#             filename = fs.save(file.name, file)
#             file_path = fs.path(filename)

#             # Use pandas to read the Excel file (.xlsx)
#             if file.name.endswith('.xlsx'):
#                 df = pd.read_excel(file_path, engine='openpyxl')

#                 # Process and save each row to the Customer model
#                 for index, row in df.iterrows():
#                     Customer.objects.create(
#                         customerID=row['customerID'],
#                         gender=row['gender'],
#                         senior_citizen=bool(row['SeniorCitizen']),
#                         partner=bool(row['Partner']),
#                         dependents=bool(row['Dependents']),
#                         tenure=int(row['tenure']),
#                         phone_service=bool(row['PhoneService']),
#                         multiple_lines=row['MultipleLines'],
#                         internet_service=row['InternetService'],
#                         online_security=row['OnlineSecurity'],
#                         online_backup=row['OnlineBackup'],
#                         device_protection=row['DeviceProtection'],
#                         tech_support=row['TechSupport'],
#                         streaming_tv=row['StreamingTV'],
#                         streaming_movies=row['StreamingMovies'],
#                         contract=row['Contract'],
#                         paperless_billing=bool(row['PaperlessBilling']),
#                         payment_method=row['PaymentMethod'],
#                         monthly_charges=float(row['MonthlyCharges']),
#                         total_charges=float(row['TotalCharges']) if pd.notnull(row['TotalCharges']) else None,
#                         churn=bool(row['Churn'])
#                     )

#                 return redirect('customer_list')  # Redirect after successful upload
#     else:
#         form = UploadFileForm()

#     return render(request,'upload_customer_data.html', {'form': form})
