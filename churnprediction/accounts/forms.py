from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from users.models import CustomUser


class CreateUserForm(UserCreationForm):

    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('organization', 'Organization'),
        ('customer', 'Customer'),
    ]
    
    role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES, required=True)
    
    class Meta:
        model = CustomUser  
        fields = ['username', 'email', 'password1', 'password2']
     



# class EmailAuthenticationForm(AuthenticationForm):
#     username = forms.EmailField(label="Email", widget=forms.EmailInput(attrs={'autofocus': True}))

#     class Meta:
#         fields = ['email', 'password']