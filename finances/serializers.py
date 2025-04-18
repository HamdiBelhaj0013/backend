from rest_framework import serializers
from .models import Donor, Transaction, BudgetAllocation, FinancialReport
from api.serializers import ProjectSerializer
from users.serializers import UserProfileSerializer
from django.utils import timezone


class DonorSerializer(serializers.ModelSerializer):
    total_donations = serializers.DecimalField(max_digits=15, decimal_places=2, read_only=True)

    class Meta:
        model = Donor
        fields = [
            'id', 'name', 'email', 'phone', 'address', 'tax_id',
            'notes', 'is_anonymous', 'total_donations',
            'created_at', 'updated_at'
        ]


class TransactionSerializer(serializers.ModelSerializer):
    project_details = ProjectSerializer(source='project', read_only=True)
    donor_details = DonorSerializer(source='donor', read_only=True)
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    verified_by_details = UserProfileSerializer(source='verified_by', read_only=True)

    class Meta:
        model = Transaction
        fields = [
            'id', 'transaction_type', 'category', 'amount', 'description',
            'date', 'document', 'project', 'project_details', 'donor', 'donor_details',
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
        fields = ['status', 'verification_notes']

    def update(self, instance, validated_data):
        instance.status = validated_data.get('status', instance.status)
        instance.verification_notes = validated_data.get('verification_notes', instance.verification_notes)

        # If the transaction is being verified, set verified_by and verification_date
        if instance.status == 'verified':
            instance.verified_by = self.context['request'].user
            instance.verification_date = timezone.now()

            # If the transaction is linked to a project, update the budget usage
            if instance.project and instance.transaction_type == 'expense':
                try:
                    budget = BudgetAllocation.objects.get(project=instance.project)
                    budget.used_amount += instance.amount
                    budget.save()
                except BudgetAllocation.DoesNotExist:
                    pass

        instance.save()
        return instance


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