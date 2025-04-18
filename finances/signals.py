from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone
from .models import Transaction, BudgetAllocation


@receiver(post_save, sender=Transaction)
def update_budget_after_transaction_verification(sender, instance, created, **kwargs):
    """
    Update budget allocation when a transaction is verified
    """
    # Only handle if status is verified and transaction is linked to a project
    if instance.status == 'verified' and instance.project:
        # Only process expense transactions
        if instance.transaction_type == 'expense':
            try:
                # Get or create budget allocation for the project
                budget, created = BudgetAllocation.objects.get_or_create(
                    project=instance.project,
                    defaults={
                        'allocated_amount': 0,
                        'used_amount': 0,
                        'created_by': instance.created_by
                    }
                )

                # If transaction was just verified, update budget
                prev_status = getattr(instance, '_prev_status', None)

                if prev_status != 'verified' and instance.status == 'verified':
                    budget.used_amount += instance.amount
                    budget.save()

            except Exception as e:
                print(f"Error updating budget: {e}")


@receiver(post_delete, sender=Transaction)
def update_budget_after_transaction_deletion(sender, instance, **kwargs):
    """
    Update budget allocation when a verified transaction is deleted
    """
    # Only handle if status was verified and transaction was linked to a project
    if instance.status == 'verified' and instance.project:
        # Only process expense transactions
        if instance.transaction_type == 'expense':
            try:
                # Reduce used amount when deleting an expense
                budget = BudgetAllocation.objects.get(project=instance.project)
                budget.used_amount = max(0, budget.used_amount - instance.amount)
                budget.save()
            except BudgetAllocation.DoesNotExist:
                pass
            except Exception as e:
                print(f"Error updating budget after deletion: {e}")


# Add pre_save handler to track status changes
@receiver(post_save, sender=Transaction)
def store_previous_status(sender, instance, **kwargs):
    """Store the previous status for reference in other signal handlers"""
    if hasattr(instance, '_prev_status'):
        delattr(instance, '_prev_status')

    try:
        prev_instance = Transaction.objects.get(pk=instance.pk)
        instance._prev_status = prev_instance.status
    except Transaction.DoesNotExist:
        instance._prev_status = None