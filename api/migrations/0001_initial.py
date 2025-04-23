from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('start_date', models.DateField()),
                ('end_date', models.DateField()),
                ('description', models.TextField(max_length=500)),
                ('budget', models.DecimalField(decimal_places=2, max_digits=10)),
                ('status', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('association', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='projects', to='users.associationaccount')),
            ],
        ),
        migrations.CreateModel(
            name='Member',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('address', models.CharField(max_length=100)),
                ('email', models.EmailField(default='example@example.com', max_length=254)),
                ('nationality', models.CharField(max_length=100)),
                ('birth_date', models.DateField()),
                ('job', models.CharField(max_length=100)),
                ('joining_date', models.DateField()),
                ('role', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('association', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='members', to='users.associationaccount')),
            ],
            options={
                'unique_together': {('name', 'association')},
            },
        ),
    ]