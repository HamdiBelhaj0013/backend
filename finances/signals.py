from django.db.models.signals import post_delete
from django.db.models.signals import post_save, pre_save
from .models import Transaction
from .models import ForeignDonationReport
from django.db.models.signals import post_save
from django.dispatch import receiver
from api.models import Project
from .models import BudgetAllocation


@receiver(post_save, sender=Project)
def create_budget_allocation_for_new_project(sender, instance, created, **kwargs):
    """
    Create a budget allocation for a new project using the project's budget value
    """
    if created:  # Only run when a project is first created
        try:
            # Check if a budget allocation already exists for this project
            existing_allocation = BudgetAllocation.objects.filter(project=instance).exists()

            if not existing_allocation:
                # Create a new budget allocation with the project's budget amount
                allocation = BudgetAllocation.objects.create(
                    project=instance,
                    allocated_amount=instance.budget,
                    used_amount=0,
                    notes=f"Budet de projet {instance.name}",
                    # Set association if available on the project
                    association=instance.association if hasattr(instance, 'association') else None
                )
                print(f"Created budget allocation of {allocation.allocated_amount} for project {instance.name}")
        except Exception as e:
            print(f"Error creating budget allocation for project {instance.id}: {str(e)}")
@receiver(post_save, sender=Transaction)
def update_budget_after_transaction_verification(sender, instance, created, **kwargs):
    """
    Update budget allocation when a transaction is verified
    """
    from django.db import transaction

    # First, handle expense transactions that are linked to projects
    if instance.status == 'verified' and instance.project and instance.transaction_type == 'expense':
        try:
            # Get or create budget allocation for the project - wrap in atomic
            with transaction.atomic():
                budget, created = BudgetAllocation.objects.get_or_create(
                    project=instance.project,
                    defaults={
                        'allocated_amount': 0,
                        'used_amount': 0,
                        'created_by': instance.created_by
                    }
                )

                # Get previous status
                prev_status = getattr(instance, '_prev_status', None)

                # Only update budget if transaction status changed to verified
                if prev_status != 'verified' and instance.status == 'verified':
                    budget.used_amount += instance.amount
                    budget.save()
                    print(f"Updated budget for project {instance.project.name}: Added expense {instance.amount}")

        except Exception as e:
            print(f"Error updating budget: {e}")

    # Process income transactions (including membership fees)
    # Log for debugging purposes
    if instance.status == 'verified' and instance.transaction_type == 'income':
        print(f"Income transaction verified: {instance.id} - {instance.amount} - Category: {instance.category}")


@receiver(post_delete, sender=Transaction)
def update_budget_after_transaction_deletion(sender, instance, **kwargs):
    """
    Update budget allocation when a verified transaction is deleted
    """
    from django.db import transaction

    # Only handle if status was verified and transaction was linked to a project
    if instance.status == 'verified' and instance.project:
        # Only process expense transactions
        if instance.transaction_type == 'expense':
            try:
                with transaction.atomic():
                    # Reduce used amount when deleting an expense
                    budget = BudgetAllocation.objects.get(project=instance.project)
                    budget.used_amount = max(0, budget.used_amount - instance.amount)
                    budget.save()
            except BudgetAllocation.DoesNotExist:
                pass
            except Exception as e:
                print(f"Error updating budget after deletion: {e}")


@receiver(pre_save, sender=Transaction)
def store_previous_status(sender, instance, **kwargs):
    """Store the previous status for reference in other signal handlers"""
    if instance.pk:  # Only for existing objects, not new ones
        try:
            prev_instance = Transaction.objects.get(pk=instance.pk)
            instance._prev_status = prev_instance.status
        except Transaction.DoesNotExist:
            instance._prev_status = None
    else:
        # For new instances
        instance._prev_status = None
@receiver(post_save, sender=Transaction)
def create_foreign_donation_report(sender, instance, created, **kwargs):
    """Create a foreign donation report when a transaction is created with a foreign donor"""
    if created and instance.transaction_type == 'income' and instance.donor:
        try:
            # Check if the donor is external (neither member nor internal)
            if not instance.donor.is_member and not instance.donor.is_internal:
                # Create a report for this transaction
                ForeignDonationReport.objects.create(transaction=instance)
                print(f"Created foreign donation report for transaction {instance.id}")
        except Exception as e:
            print(f"Error creating foreign donation report: {str(e)}")