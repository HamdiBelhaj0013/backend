from django.db.models.signals import post_delete
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from .models import Transaction, BudgetAllocation


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