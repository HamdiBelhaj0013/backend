from rest_framework import serializers
from .models import Donor, Transaction, BudgetAllocation, FinancialReport
from api.serializers import ProjectSerializer
from django.utils import timezone
from .models import Donor
from rest_framework import serializers
from users.serializers import UserProfileSerializer
from django.apps import apps


class DonorSerializer(serializers.ModelSerializer):

    total_donations = serializers.SerializerMethodField()
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    member_details = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Donor
        fields = [
            'id', 'name', 'email', 'phone', 'address',
            'tax_id', 'notes', 'is_anonymous',
            'is_member', 'is_internal', 'member_id', 'member_details',
            'is_active', 'created_at', 'updated_at', 'total_donations',
            'created_by', 'created_by_details', 'association'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'total_donations',
            'created_by', 'created_by_details', 'association'
        ]

    def get_total_donations(self, obj):
        # This calls the total_donations property on the model
        return obj.total_donations

    def get_member_details(self, obj):
        """Return member details if this donor is a member"""
        if not obj.is_member or not obj.member_id:
            return None

        # Get the Member model dynamically to avoid circular imports
        Member = apps.get_model('api', 'Member')

        try:
            member = Member.objects.get(id=obj.member_id)
            return {
                'id': member.id,
                'name': member.name if hasattr(member, 'name') else member.full_name if hasattr(member,
                                                                                                'full_name') else member.username,
                'email': member.email if hasattr(member, 'email') else None,
                'role': member.role if hasattr(member, 'role') else None
            }
        except Member.DoesNotExist:
            return None

    def validate(self, data):
        """
        Validate the donor data.
        - If is_member is True, ensure is_internal is False
        - If is_internal is True, ensure is_member is False
        - If is_member is True, require member_id
        """
        is_member = data.get('is_member', False)
        is_internal = data.get('is_internal', False)
        member_id = data.get('member_id')

        if is_member and is_internal:
            raise serializers.ValidationError("A donor cannot be both a member and internal at the same time")

        if is_member and not member_id:
            raise serializers.ValidationError("A member donor must have a member_id")

        if not is_member and member_id:
            data['member_id'] = None  # Clear member_id if not a member

        # If this is a member donor, we can set name based on member details
        if is_member and member_id and not data.get('name'):
            Member = apps.get_model('api', 'Member')
            try:
                member = Member.objects.get(id=member_id)
                data['name'] = member.name if hasattr(member, 'name') else member.full_name if hasattr(member,
                                                                                                       'full_name') else member.username
                # Also set email if available and not already set
                if not data.get('email') and hasattr(member, 'email') and member.email:
                    data['email'] = member.email
            except Member.DoesNotExist:
                pass

        return data

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
        # Store the initial status before any changes
        initial_status = instance.status

        # Update the transaction with the validated data
        instance.status = validated_data.get('status', instance.status)
        instance.verification_notes = validated_data.get('verification_notes', instance.verification_notes)

        # Allow updating budget_allocation during verification if it's not already set
        if 'budget_allocation' in validated_data and not instance.budget_allocation:
            instance.budget_allocation = validated_data.get('budget_allocation')

        # If the transaction is being verified for the first time, set verified_by and verification_date
        if instance.status == 'verified' and initial_status != 'verified':
            instance.verified_by = self.context['request'].user
            instance.verification_date = timezone.now()

            print(f"Transaction {instance.id} verified by {instance.verified_by.email}")
            print(f"  Initial status: {initial_status}, New status: {instance.status}")

            # Note: budget allocation is handled by signals, so we don't need to manually
            # update the budget here, which could lead to double-counting

        # If the transaction is being rejected after being verified, need to update budget
        if instance.status == 'rejected' and initial_status == 'verified':
            # If the transaction has a budget allocation and is an expense, update the budget
            if instance.budget_allocation and instance.transaction_type == 'expense':
                try:
                    budget = instance.budget_allocation
                    # Subtract the amount from used_amount since it was added when verified
                    budget.used_amount = max(0, budget.used_amount - instance.amount)
                    budget.save()
                    print(f"Transaction rejected: Removed {instance.amount} from budget {budget.id}")
                except Exception as e:
                    print(f"Error updating budget on rejection: {e}")

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