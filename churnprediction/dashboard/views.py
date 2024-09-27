from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import ChurnPrediction

# Create your views here.
@login_required(login_url='/accounts/login/')
def dashboard_view(request):
    predictions = ChurnPrediction.objects.all()

    return render(request, 'dashboard/churn_dashboard.html', {'predictions': predictions})


