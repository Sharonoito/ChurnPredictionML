from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class CustomerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # Link to Django's built-in User model
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    signup_date = models.DateField(auto_now_add=True)
    # Add any additional fields relevant to your business logic

    def __str__(self):
        return self.user.username  # Returns the username of the customer
