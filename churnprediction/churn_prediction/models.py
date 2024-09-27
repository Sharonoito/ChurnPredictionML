from django.db import models

class Customer(models.Model):
    customerID = models.CharField(max_length=50, unique=True)
    gender = models.CharField(max_length=10)
    senior_citizen = models.BooleanField()
    partner = models.BooleanField()
    dependents = models.BooleanField()
    tenure = models.IntegerField()
    phone_service = models.BooleanField()
    multiple_lines = models.CharField(max_length=20)
    internet_service = models.CharField(max_length=20)
    online_security = models.CharField(max_length=20)
    online_backup = models.CharField(max_length=20)
    device_protection = models.CharField(max_length=20)
    tech_support = models.CharField(max_length=20)
    streaming_tv = models.CharField(max_length=20)
    streaming_movies = models.CharField(max_length=20)
    contract = models.CharField(max_length=20)
    paperless_billing = models.BooleanField()
    payment_method = models.CharField(max_length=50)
    monthly_charges = models.DecimalField(max_digits=10, decimal_places=2)
    total_charges = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    churn = models.BooleanField()

    def __str__(self):
        return self.customerID


class ChurnPrediction(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='churn_predictions')
    churn_probability = models.FloatField()  # Probability of churn (0 to 1)
    prediction_label = models.CharField(max_length=50)  # E.g., "High Risk", "Low Risk", etc.
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.customer.name} - {self.prediction_label} ({self.churn_probability})"


class Interaction(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='interactions')
    interaction_type = models.CharField(max_length=100)  # E.g., "Purchase", "Support Call"
    interaction_date = models.DateTimeField(auto_now_add=True)
    details = models.TextField(blank=True, null=True)  # Additional interaction details

    def __str__(self):
        return f"{self.customer.name} - {self.interaction_type}"


class Demographics(models.Model):
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE, related_name='demographics')
    age = models.IntegerField(blank=True, null=True)
    gender = models.CharField(max_length=10, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.customer.name}'s Demographics"

