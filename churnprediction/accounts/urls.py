from django.urls import path
from django.contrib.auth import views as auth_views
from . import views


urlpatterns = [
    
path('register/', views.register_page, name="register"),
path('login/', views.login_page, name="login"),
path('logout/', views.logout_user, name="logout"),
path('profile/', views.profile_page, name="profile"),

path('admin_dashboard/', views.admin_dashboard, name="admin_dashboard"),
path('dashboard/', views.organization_dashboard, name="organization_dashboard"),
path('customer_dashboard/', views.customer_dashboard, name="customer_dashboard"),
    

path('reset_password/',
     auth_views.PasswordResetView.as_view(template_name="password_reset.html"),
     name="reset_password"),
path('reset_password_sent/', 
        auth_views.PasswordResetDoneView.as_view(template_name="password_reset_sent.html"), 
        name="password_reset_done"),

    path('reset/<uidb64>/<token>/',
     auth_views.PasswordResetConfirmView.as_view(template_name="password_reset_form.html"), 
     name="password_reset_confirm"),

    path('reset_password_complete/', 
        auth_views.PasswordResetCompleteView.as_view(template_name="password_reset_done.html"), 
        name="password_reset_complete"),

]