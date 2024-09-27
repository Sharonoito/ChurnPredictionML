from django.db import models
# from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager



from django.conf import settings

# Create your models here.


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, role='customer', **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, role=role, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, role='admin', **extra_fields)   
    

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
    ('admin', 'Admin'),
    ('organization', 'Organization'),
    ('customer', 'Customer'),
    )

    role = models.CharField(max_length=15, choices=ROLE_CHOICES, default='customer') 
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email
        
    
class CustomerProfile(models.Model):
    # user = models.OneToOneField(User, on_delete=models.CASCADE) 
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    signup_date = models.DateField(auto_now_add=True)
    # Add any additional fields relevant to your business logic

    def __str__(self):
        return self.user.username  # Returns the username of the customer

