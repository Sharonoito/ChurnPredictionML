from django.contrib import admin
from .models import Customer, ChurnPrediction, Interaction, Demographics

admin.site.register(Customer)
admin.site.register(ChurnPrediction)
admin.site.register(Interaction)
admin.site.register(Demographics)

