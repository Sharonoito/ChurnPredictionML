# Generated by Django 5.1.1 on 2024-09-11 20:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChurnPrediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prediction_date', models.DateTimeField(auto_now_add=True)),
                ('churn_probability', models.DecimalField(decimal_places=2, max_digits=5)),
                ('churn_label', models.BooleanField()),
                ('customer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.customerprofile')),
            ],
        ),
        migrations.CreateModel(
            name='CustomerInteraction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('interaction_date', models.DateTimeField()),
                ('interaction_type', models.CharField(max_length=255)),
                ('notes', models.TextField()),
                ('customer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.customerprofile')),
            ],
        ),
    ]
