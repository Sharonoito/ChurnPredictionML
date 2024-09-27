from django.shortcuts import render
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required(login_url='/accounts/login/')
def dashboard_view(request):
    # Your dashboard logic here
    return render(request, 'admin_dashboard/admindashboard.html')
