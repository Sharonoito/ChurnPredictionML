from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import *
from .forms import CreateUserForm
from userprofile.models import UserProfile  
from .decorators import role_required 


# Register page
def register_page(request):
    if request.user.is_authenticated:
        return redirect('login')  # Redirect to dashboard if already authenticated
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                user = form.save()
                UserProfile.objects.create(user=user)
                messages.success(request, 'Account was created successfully.')
                return redirect('login')  # Redirect to login after successful registration
        context = {'form': form}
        return render(request, 'register.html', context)


# Login page
def login_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Redirect to dashboard if already logged in

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')  # Redirect to a generic dashboard, handled below
        else:
            messages.info(request, 'Email or password is incorrect')

    return render(request, 'login.html')

# Logout function
def logout_user(request):
    logout(request)
    return redirect('login')

# Profile page
def profile_page(request):
    if not request.user.is_authenticated:
        return redirect('login')  # Redirect to login if not authenticated

    try:
        user_profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        user_profile = None

    context = {'user_profile': user_profile}
    return render(request, 'profile.html', context)

# Role-based dashboards using the @role_required decorator
@role_required(allowed_roles=['admin'])
def admin_dashboard(request):
    return render(request, 'admindashboard.html')

@role_required(allowed_roles=['organization'])
def organization_dashboard(request):
    return render(request, 'dashboard.html')

@role_required(allowed_roles=['customer'])
def customer_dashboard(request):
    return render(request, 'customerdashboard.html')

# Generic dashboard to redirect users based on role
def dashboard(request):
    if request.user.role == 'admin':
        return redirect('admin_dashboard')  # Redirect to admin dashboard
    elif request.user.role == 'organization':
        return redirect('organization_dashboard')  # Redirect to organization dashboard
    elif request.user.role == 'customer':
        return redirect('customer_dashboard')  # Redirect to customer dashboard
    else:
        return render(request, 'dashboard.html')  # Fallback for any role without a specific dashboard

        

