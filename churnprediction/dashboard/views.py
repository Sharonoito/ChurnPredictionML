# views.py
from django.shortcuts import render

def dashboard(request):
    # Example data for the chart
    bar_chart_data = [20, 30, 40, 50, 60, 70]
    bar_chart_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    line_chart_data = [50, 40, 60, 80, 70, 60]
    line_chart_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    context = {
        'bar_chart_data': bar_chart_data,
        'bar_chart_labels': bar_chart_labels,
        'line_chart_data': line_chart_data,
        'line_chart_labels': line_chart_labels
    }
    return render(request, 'dashboard.html', context)

# from django.shortcuts import render
# from django.contrib.auth.decorators import login_required
# from .models import ChurnPrediction

# @login_required(login_url='/accounts/login/')
# def dashboard_view(request):
#     predictions = ChurnPrediction.objects.all()

#     return render(request, 'dashboard/churn_dashboard.html', {'predictions': predictions})


