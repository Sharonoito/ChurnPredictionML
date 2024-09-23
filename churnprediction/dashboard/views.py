from django.shortcuts import render
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required(login_url='/accounts/login/')
def dashboard_view(request):
    # Your dashboard logic here
    return render(request, 'dashboard/dashboard.html')

# from .models import ChurnPrediction, AnalyticsSummary

# def dashboard_view(request):
#     # Example of fetching the latest predictions
#     recent_predictions = ChurnPrediction.objects.order_by('-prediction_date')[:10]

#     # Example of fetching the latest analytics summary
#     latest_summary = AnalyticsSummary.objects.latest('summary_date')

#     context = {
#         'recent_predictions': recent_predictions,
#         'latest_summary': latest_summary,
#     }
#     return render(request, 'dashboard.html', context)
