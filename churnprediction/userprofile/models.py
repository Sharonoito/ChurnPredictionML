from django.db import models
from django.conf import settings

# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')])
    email = models.EmailField(null=True)
    telephone_number = models.CharField(max_length=20,null=True)
    custom_fields = models.JSONField(blank=True, null=True)  # Use Django's built-in JSONField
    address = models.TextField(blank=True, null=True)
    organization = models.CharField(max_length=100, blank=True, null=True)  # Example field


    def __str__(self):
        return self.user.username
    