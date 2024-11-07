# churn_prediction/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('upload-customer-data/', views.upload_customer_data, name='upload_customer_data'),
    path('customer-list/', views.customer_list, name='customer_list'), 
    path('prediction-results/',views.prediction_results,name='prediction_results'), 
    path('prediction-reports/',views.prediction_reports,name='prediction_reports'),

]


