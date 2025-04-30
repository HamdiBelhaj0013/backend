from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone

from finances.models import Transaction
from .services import NotificationService


@receiver(post_save, sender=Transaction)
def handle_transaction_save(sender, instance, created, **kwargs):
    """
    Handle transaction model save events to send appropriate notifications
    """
    # For new transactions (creation)
    if created:
        # Process this transaction - send notifications or schedule letters if needed
        NotificationService.process_critical_transaction(
            transaction=instance,
            action_type="create"
        )
    else:
        # For updates to existing transactions
        NotificationService.process_critical_transaction(
            transaction=instance,
            action_type="update"
        )


@receiver(post_delete, sender=Transaction)
def handle_transaction_delete(sender, instance, **kwargs):
    """
    Handle transaction model delete events to send appropriate notifications
    """
    # Process the deleted transaction
    NotificationService.process_critical_transaction(
        transaction=instance,
        action_type="delete"
    )