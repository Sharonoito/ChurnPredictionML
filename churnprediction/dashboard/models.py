# from django.db import models

# # Create your models here.
# from django.db import models
# from django.contrib.auth.models import User

# # Model to store the churn predictions
# class ChurnPrediction(models.Model):
#     customer = models.ForeignKey('CustomerProfile', on_delete=models.CASCADE)
#     prediction_date = models.DateTimeField(auto_now_add=True)
#     churn_probability = models.DecimalField(max_digits=5, decimal_places=2)
#     churn_label = models.BooleanField()  # True for churn, False for retain

#     def __str__(self):
#         return f"Prediction for {self.customer} on {self.prediction_date}"

# # Model to track customer interactions
# class CustomerInteraction(models.Model):
#     customer = models.ForeignKey('CustomerProfile', on_delete=models.CASCADE)
#     interaction_date = models.DateTimeField()
#     interaction_type = models.CharField(max_length=255)
#     notes = models.TextField()

#     def __str__(self):
#         return f"{self.interaction_type} with {self.customer}"

# # Model for historical data trends (e.g., churn rates, customer stats)
# class HistoricalData(models.Model):
#     date = models.DateField()
#     total_customers = models.IntegerField()
#     churned_customers = models.IntegerField()
#     retained_customers = models.IntegerField()

#     def churn_rate(self):
#         return (self.churned_customers / self.total_customers) * 100

#     def __str__(self):
#         return f"Data for {self.date}"

# # Model to store aggregated analytics summaries for quick access on the dashboard
# class AnalyticsSummary(models.Model):
#     summary_date = models.DateTimeField(auto_now_add=True)
#     total_customers = models.IntegerField()
#     churn_rate = models.DecimalField(max_digits=5, decimal_places=2)
#     avg_prediction_accuracy = models.DecimalField(max_digits=5, decimal_places=2)

#     def __str__(self):
#         return f"Summary for {self.summary_date}"
from django.db import models

class ChurnPrediction(models.Model):
    customer = models.ForeignKey('users.CustomerProfile', on_delete=models.CASCADE)
    prediction_date = models.DateTimeField(auto_now_add=True)
    churn_probability = models.DecimalField(max_digits=5, decimal_places=2)
    churn_label = models.BooleanField()

    def __str__(self):
        return f"Prediction for {self.customer} on {self.prediction_date}"

class CustomerInteraction(models.Model):
    customer = models.ForeignKey('users.CustomerProfile', on_delete=models.CASCADE)
    interaction_date = models.DateTimeField()
    interaction_type = models.CharField(max_length=255)
    notes = models.TextField()

    def __str__(self):
        return f"{self.interaction_type} with {self.customer}"