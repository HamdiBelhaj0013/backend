from rest_framework import serializers
from .models import Donor, Transaction, BudgetAllocation, FinancialReport
from api.serializers import ProjectSerializer
from users.serializers import UserProfileSerializer
from django.utils import timezone

# Add this to your existing serializers.py file
from rest_framework import serializers
from .models import Donor

# Update your DonorSerializer.py file with this code
from rest_framework import serializers
from .models import Donor


class DonorSerializer(serializers.ModelSerializer):
    # Using SerializerMethodField for read-only properties
    total_donations = serializers.SerializerMethodField()

    class Meta:
        model = Donor
        fields = [
            'id', 'name', 'email', 'phone', 'address',
            'tax_id', 'notes', 'is_anonymous',
            'created_at', 'updated_at', 'total_donations'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'total_donations']

    def get_total_donations(self, obj):
        # This calls the total_donations property on the model
        return obj.total_donations


class BudgetAllocationSerializer(serializers.ModelSerializer):
    project_details = ProjectSerializer(source='project', read_only=True)
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    remaining_amount = serializers.DecimalField(max_digits=15, decimal_places=2, read_only=True)
    utilization_percentage = serializers.FloatField(read_only=True)

    class Meta:
        model = BudgetAllocation
        fields = [
            'id', 'project', 'project_details', 'allocated_amount',
            'used_amount', 'remaining_amount', 'utilization_percentage',
            'notes', 'created_by', 'created_by_details',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['used_amount']

    def create(self, validated_data):
        # Add the current user as the creator
        validated_data['created_by'] = self.context['request'].user
        return super().create(validated_data)


class TransactionSerializer(serializers.ModelSerializer):
    project_details = ProjectSerializer(source='project', read_only=True)
    donor_details = DonorSerializer(source='donor', read_only=True)
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    verified_by_details = UserProfileSerializer(source='verified_by', read_only=True)
    budget_allocation_details = BudgetAllocationSerializer(source='budget_allocation', read_only=True)

    class Meta:
        model = Transaction
        fields = [
            'id', 'transaction_type', 'category', 'amount', 'description',
            'date', 'document', 'project', 'project_details', 'donor', 'donor_details',
            'budget_allocation', 'budget_allocation_details',
            'reference_number', 'status', 'verified_by', 'verified_by_details',
            'verification_date', 'verification_notes', 'created_by', 'created_by_details',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['verified_by', 'verification_date']

    def create(self, validated_data):
        # Add the current user as the creator
        validated_data['created_by'] = self.context['request'].user
        return super().create(validated_data)


class TransactionVerificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = ['status', 'verification_notes', 'budget_allocation']

    def update(self, instance, validated_data):
        instance.status = validated_data.get('status', instance.status)
        instance.verification_notes = validated_data.get('verification_notes', instance.verification_notes)

        # Allow updating budget_allocation during verification if it's not already set
        if 'budget_allocation' in validated_data and not instance.budget_allocation:
            instance.budget_allocation = validated_data.get('budget_allocation')

        # If the transaction is being verified, set verified_by and verification_date
        if instance.status == 'verified':
            instance.verified_by = self.context['request'].user
            instance.verification_date = timezone.now()

            # If the transaction is linked to a budget allocation and is an expense, update the budget usage
            if instance.budget_allocation and instance.transaction_type == 'expense':
                try:
                    budget = instance.budget_allocation
                    budget.used_amount += instance.amount
                    budget.save()
                except Exception as e:
                    # Log the error but don't prevent verification
                    print(f"Error updating budget: {e}")

        instance.save()
        return instance


class FinancialReportSerializer(serializers.ModelSerializer):
    generated_by_details = UserProfileSerializer(source='generated_by', read_only=True)

    class Meta:
        model = FinancialReport
        fields = [
            'id', 'report_type', 'title', 'start_date', 'end_date',
            'status', 'report_file', 'generated_by', 'generated_by_details',
            'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = ['generated_by']

    def create(self, validated_data):
        # Add the current user as the generator
        validated_data['generated_by'] = self.context['request'].user
        return super().create(validated_data)


class FinancialStatisticsSerializer(serializers.Serializer):
    total_income = serializers.DecimalField(max_digits=15, decimal_places=2)
    total_expenses = serializers.DecimalField(max_digits=15, decimal_places=2)
    net_balance = serializers.DecimalField(max_digits=15, decimal_places=2)
    total_donations = serializers.DecimalField(max_digits=15, decimal_places=2)
    total_project_expenses = serializers.DecimalField(max_digits=15, decimal_places=2)
    income_by_category = serializers.DictField(child=serializers.DecimalField(max_digits=15, decimal_places=2))
    expenses_by_category = serializers.DictField(child=serializers.DecimalField(max_digits=15, decimal_places=2))
    project_budget_utilization = serializers.ListField(child=serializers.DictField())
    recent_transactions = TransactionSerializer(many=True)