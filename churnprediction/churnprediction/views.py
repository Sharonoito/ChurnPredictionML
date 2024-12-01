# churnprediction/views.py

from django.shortcuts import render

def dashboard_view(request):
    return render(request, 'dashboard.html')

def account_view(request):
    return render(request,'login.html')
