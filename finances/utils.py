import os
import tempfile
from datetime import timedelta
from decimal import Decimal
from django.db.models import Sum, Q
from django.utils import timezone
import io
import csv
from .models import Transaction, BudgetAllocation, ForeignDonationReport


def generate_foreign_donation_journal_publication(report):
    """Generate a journal publication text for foreign donation report with error handling"""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Starting journal publication generation for report {report.id}")

    try:
        # Get transaction and donor information
        transaction = report.transaction
        if not transaction:
            msg = f"Transaction not found for report {report.id}"
            logger.error(msg)
            raise ValueError(msg)

        donor = transaction.donor
        if not donor:
            msg = f"Donor not found for transaction {transaction.id}"
            logger.error(msg)
            raise ValueError(msg)

        # Get association data
        association = None
        if hasattr(transaction, 'association') and transaction.association:
            association = transaction.association
        elif hasattr(transaction, 'project') and transaction.project and hasattr(transaction.project, 'association'):
            association = transaction.project.association

        # Format donor and transaction information
        donor_name = donor.name if donor and hasattr(donor, 'name') else "Non spécifié"
        # Anonymize donor if requested
        if donor.is_anonymous:
            donor_name = "Donateur Anonyme"

        # Format transaction amount safely
        try:
            transaction_amount = f"{transaction.amount} TND" if hasattr(transaction, 'amount') else "Non spécifié"
        except Exception as e:
            logger.warning(f"Error formatting transaction amount: {e}")
            transaction_amount = "Non spécifié"

        # Format transaction date safely
        try:
            transaction_date = transaction.date.strftime("%d/%m/%Y") if hasattr(transaction,
                                                                                'date') and transaction.date else "Non spécifié"
        except Exception as e:
            logger.warning(f"Error formatting transaction date: {e}")
            transaction_date = "Non spécifié"

        # Get association details or use defaults
        if association:
            association_name = association.name if hasattr(association, 'name') else "Association"
            association_president = (
                association.president_name if hasattr(association, 'president_name')
                else (association.president.get_full_name() if hasattr(association, 'president') else "")
            )
        else:
            association_name = "Votre Association"
            association_president = "Le Président de l'Association"

        # Generate publication text
        publication_text = f"""
AVIS DE DON D'ORIGINE ETRANGERE

Conformément aux dispositions du décret-loi n° 2011-88 du 24 septembre 2011, portant organisation des associations, 
{association_name} déclare avoir reçu un don d'origine étrangère de {transaction_amount} en date du {transaction_date}.

Le don, versé par {donor_name}, a été déclaré au Premier Ministre conformément à la loi. Les fonds seront utilisés 
dans le cadre des activités de l'association conformément aux objectifs énoncés dans ses statuts.

Pour {association_name}
{association_president}
"""

        # Save the generated text to the report
        report.journal_publication_text = publication_text
        report.save(update_fields=['journal_publication_text'])

        logger.info(f"Journal publication text generated for report {report.id}")
        return report

    except Exception as e:
        import traceback
        logger.error(f"Error generating journal publication: {str(e)}")
        logger.error(traceback.format_exc())

        # Add error info to report
        if hasattr(report, 'notes'):
            report.notes = (report.notes or '') + f"\nError generating journal publication: {str(e)}"
            report.save(update_fields=['notes'])

        raise


# Add this to your utils.py file after the existing get_financial_statistics function

def get_financial_statistics(start_date, end_date, association=None):
    """
    Get financial statistics for a date range, optionally filtered by association
    """
    from .models import Transaction, BudgetAllocation
    from django.db.models import Sum, Q
    from decimal import Decimal
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Calculating financial statistics for period {start_date} to {end_date}")

    # Base query for verified transactions in the date range
    transaction_query = {
        'status': 'verified',
        'date__gte': start_date,
        'date__lte': end_date,
    }

    # Add association filter if provided
    if association:
        # Include both direct association and project association
        transactions = Transaction.objects.filter(
            Q(association=association) |
            Q(project__association=association),
            **transaction_query
        ).distinct()
    else:
        transactions = Transaction.objects.filter(**transaction_query)

    # Calculate total income and expenses
    income_transactions = transactions.filter(transaction_type='income')
    expense_transactions = transactions.filter(transaction_type='expense')

    total_income = income_transactions.aggregate(total=Sum('amount'))['total'] or Decimal('0')
    total_expenses = expense_transactions.aggregate(total=Sum('amount'))['total'] or Decimal('0')
    net_balance = total_income - total_expenses

    # Check for membership fee transactions specifically
    membership_transactions = income_transactions.filter(category='membership_fee')
    membership_count = membership_transactions.count()
    logger.info(f"Found {membership_count} membership fee transactions in date range")

    # Debug all income transaction categories
    categories_debug = income_transactions.values_list('category', flat=True).distinct()
    logger.info(f"Income categories found: {list(categories_debug)}")

    # Calculate statistics by category
    income_by_category = {}
    expenses_by_category = {}

    # Income categories
    categories = transactions.filter(transaction_type='income').values_list('category', flat=True).distinct()
    for category in categories:
        category_total = transactions.filter(
            transaction_type='income',
            category=category
        ).aggregate(total=Sum('amount'))['total'] or Decimal('0')
        income_by_category[category] = category_total
        logger.info(f"Category '{category}' total: {category_total}")

    # Expense categories
    categories = transactions.filter(transaction_type='expense').values_list('category', flat=True).distinct()
    for category in categories:
        category_total = transactions.filter(
            transaction_type='expense',
            category=category
        ).aggregate(total=Sum('amount'))['total'] or Decimal('0')
        expenses_by_category[category] = category_total

    # Calculate total donations
    total_donations = transactions.filter(
        transaction_type='income',
        category='donation'
    ).aggregate(total=Sum('amount'))['total'] or Decimal('0')

    # Calculate membership fees with explicit debugging
    total_membership_fees = transactions.filter(
        transaction_type='income',
        category='membership_fee'
    ).aggregate(total=Sum('amount'))['total'] or Decimal('0')

    logger.info(f"Calculated total_membership_fees: {total_membership_fees}")

    # Check for alternate spellings or categories that might contain membership fees
    alternate_categories = ['membership', 'cotisation', 'cotisations', 'member_fee', 'member fee']
    for alt_category in alternate_categories:
        alt_total = transactions.filter(
            transaction_type='income',
            category__icontains=alt_category
        ).aggregate(total=Sum('amount'))['total'] or Decimal('0')
        if alt_total > 0:
            logger.info(f"Found potential membership fees in alternate category '{alt_category}': {alt_total}")

    # Calculate total project expenses
    total_project_expenses = transactions.filter(
        transaction_type='expense',
        category='project_expense'
    ).aggregate(total=Sum('amount'))['total'] or Decimal('0')

    # Get project budget utilization
    budget_query = BudgetAllocation.objects.all()
    if association:
        budget_query = budget_query.filter(
            Q(association=association) |
            Q(project__association=association)
        ).distinct()

    project_budget_utilization = []
    for budget in budget_query:
        # Calculate period expenses for this project
        period_expenses = transactions.filter(
            project=budget.project,
            transaction_type='expense'
        ).aggregate(total=Sum('amount'))['total'] or Decimal('0')

        project_budget_utilization.append({
            'project': budget.project.name,
            'allocated': float(budget.allocated_amount),
            'used': float(budget.used_amount),
            'remaining': float(budget.remaining_amount),
            'utilization': float(budget.utilization_percentage),
            'period_expenses': float(period_expenses)
        })

    # Get recent transactions
    recent_transactions = transactions.order_by('-date')[:10]

    stats = {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_balance': net_balance,
        'total_donations': total_donations,
        'total_membership_fees': total_membership_fees,
        'total_project_expenses': total_project_expenses,
        'income_by_category': income_by_category,
        'expenses_by_category': expenses_by_category,
        'project_budget_utilization': project_budget_utilization,
        'recent_transactions': recent_transactions,
        'start_date': start_date,
        'end_date': end_date
    }

    logger.info(f"Returning statistics with membership_fees={total_membership_fees}")
    return stats


def export_transactions_to_csv(queryset, output_file=None):
    """
    Export transactions to CSV file

    Args:
        queryset: Transaction queryset to export
        output_file: File-like object to write to, or None to return string

    Returns:
        CSV content as string if output_file is None, otherwise None
    """
    # Define fields to export
    fieldnames = [
        'id', 'transaction_type', 'category', 'amount',
        'description', 'date', 'project', 'donor',
        'reference_number', 'status'
    ]

    # Create CSV buffer
    if output_file is None:
        output = io.StringIO()
    else:
        output = output_file

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    # Write data
    for transaction in queryset:
        writer.writerow({
            'id': transaction.id,
            'transaction_type': transaction.get_transaction_type_display(),
            'category': transaction.get_category_display(),
            'amount': transaction.amount,
            'description': transaction.description,
            'date': transaction.date,
            'project': transaction.project.name if transaction.project else '',
            'donor': transaction.donor.name if transaction.donor else '',
            'reference_number': transaction.reference_number or '',
            'status': transaction.get_status_display()
        })

    # Return CSV content if no output file provided
    if output_file is None:
        return output.getvalue()


def generate_foreign_donation_letter(report):
    """Generate a letter to the Prime Ministry about a foreign donation with enhanced error handling"""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Starting letter generation for report {report.id}")

    try:
        # Required imports
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from django.utils import timezone
        import io
        from django.core.files.base import ContentFile

        # Get transaction and donor information
        transaction = report.transaction
        if not transaction:
            msg = f"Transaction not found for report {report.id}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Found transaction: {transaction.id}, type: {transaction.transaction_type}")

        donor = transaction.donor
        if not donor:
            msg = f"Donor not found for transaction {transaction.id}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Found donor: {donor.id}, name: {donor.name if hasattr(donor, 'name') else 'Unknown'}")

        # Get association data from the transaction
        association = None
        if hasattr(transaction, 'association') and transaction.association:
            association = transaction.association
            logger.info(f"Found association directly from transaction: {association.id}")
        elif hasattr(transaction, 'project') and transaction.project and hasattr(transaction.project, 'association'):
            association = transaction.project.association
            logger.info(f"Found association from project: {association.id}")

        # Set up document
        buffer = io.BytesIO()
        logger.info("Creating PDF document")

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        title_style = styles['Title']
        heading_style = styles['Heading1']
        subheading_style = styles['Heading2']

        # Get association details or use defaults if not available
        if association:
            logger.info(
                f"Processing association details for {association.name if hasattr(association, 'name') else 'Unknown'}")
            association_name = association.name if hasattr(association, 'name') else "Association"
            association_address1 = (
                association.address_line1 if hasattr(association, 'address_line1')
                else association.address if hasattr(association, 'address')
                else "Tunisia"
            )
            association_address2 = association.address_line2 if hasattr(association, 'address_line2') else ""
            association_phone = association.phone if hasattr(association, 'phone') else ""
            association_email = association.email if hasattr(association, 'email') else ""
            president_name = (
                association.president_name if hasattr(association, 'president_name')
                else (association.president.get_full_name() if hasattr(association, 'president') else "")
            )
        else:
            # Use generic placeholders if association is not available
            logger.warning("No association found, using default values")
            association_name = "Your Association"
            association_address1 = "Association Address"
            association_address2 = "Tunisia"
            association_phone = ""
            association_email = ""
            president_name = "Association President"

        # Add organization header/letterhead
        logger.info("Building letter content")
        elements.append(Paragraph(association_name, title_style))
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(Paragraph(association_address1, normal_style))
        if association_address2:
            elements.append(Paragraph(association_address2, normal_style))
        if association_phone:
            elements.append(Paragraph(f"Phone: {association_phone}", normal_style))
        if association_email:
            elements.append(Paragraph(f"Email: {association_email}", normal_style))
        elements.append(Spacer(1, 1 * cm))

        # Date
        today = timezone.now().date().strftime("%d/%m/%Y")
        elements.append(Paragraph(f"Date: {today}", normal_style))
        elements.append(Spacer(1, 0.5 * cm))

        # Recipient
        elements.append(Paragraph("À l'attention de Monsieur le Premier Ministre", heading_style))
        elements.append(Paragraph("Gouvernement de la République Tunisienne", normal_style))
        elements.append(Paragraph("Place du Gouvernement - La Kasbah", normal_style))
        elements.append(Paragraph("1020 Tunis", normal_style))
        elements.append(Spacer(1, 1 * cm))

        # Subject
        elements.append(Paragraph("Objet: Déclaration de don d'origine étrangère", subheading_style))
        elements.append(Spacer(1, 0.5 * cm))

        # Greeting
        elements.append(Paragraph("Monsieur le Premier Ministre,", normal_style))
        elements.append(Spacer(1, 0.5 * cm))

        # Body
        letter_body = """
        Conformément aux dispositions légales régissant les associations en Tunisie, nous avons l'honneur de vous informer que notre association a reçu un don d'origine étrangère selon les détails suivants:
        """
        elements.append(Paragraph(letter_body, normal_style))
        elements.append(Spacer(1, 0.5 * cm))

        # Safely get donor and transaction information
        logger.info("Adding transaction details to letter")
        donor_name = donor.name if donor and hasattr(donor, 'name') else "Non spécifié"
        donor_address = donor.address if donor and hasattr(donor, 'address') else "Non spécifié"

        # Convert amount to string safely
        try:
            transaction_amount = f"{transaction.amount} TND" if hasattr(transaction, 'amount') else "Non spécifié"
        except Exception as e:
            logger.warning(f"Error formatting transaction amount: {e}")
            transaction_amount = "Non spécifié"

        # Format date safely
        try:
            transaction_date = transaction.date.strftime("%d/%m/%Y") if hasattr(transaction,
                                                                                'date') and transaction.date else "Non spécifié"
        except Exception as e:
            logger.warning(f"Error formatting transaction date: {e}")
            transaction_date = "Non spécifié"

        transaction_ref = transaction.reference_number or "Non spécifié"
        transaction_desc = transaction.description or "Non spécifié"

        # Log data for debugging
        logger.info(f"Donor: {donor_name}, Amount: {transaction_amount}, Date: {transaction_date}")

        # Donation details
        data = [
            ["Donateur:", donor_name],
            ["Pays d'origine:", donor_address],
            ["Montant reçu:", transaction_amount],
            ["Date de réception:", transaction_date],
            ["Numéro de référence:", transaction_ref],
            ["Description:", transaction_desc]
        ]

        # Create the table
        table = Table(data, colWidths=[4 * cm, 10 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * cm))

        # Additional information
        additional_info = """
        Nous confirmons que ces fonds seront utilisés conformément aux objectifs de notre association et à la législation en vigueur.

        Nous vous informons également que, conformément à la loi, cette déclaration sera publiée dans un journal quotidien diffusé sur le territoire national dans un délai maximum de trente jours à compter de la date de réception du don.
        """
        elements.append(Paragraph(additional_info, normal_style))
        elements.append(Spacer(1, 1 * cm))

        # Closing
        elements.append(
            Paragraph("Veuillez agréer, Monsieur le Premier Ministre, l'expression de notre haute considération.",
                      normal_style))
        elements.append(Spacer(1, 1 * cm))

        # Signature
        elements.append(Paragraph("Le Président de l'Association", normal_style))
        elements.append(Spacer(1, 2 * cm))
        elements.append(Paragraph("____________________", normal_style))
        elements.append(Paragraph(president_name, normal_style))
        elements.append(Paragraph("[Cachet de l'Association]", normal_style))

        # Build PDF
        logger.info("Building PDF document")
        doc.build(elements)

        # Create a filename with transaction ID and date
        filename = f"foreign_donation_letter_tx{transaction.id}_{timezone.now().strftime('%Y%m%d')}.pdf"

        # Save the file to the model
        logger.info(f"Saving PDF file as {filename}")

        # Check if the report already has a letter file
        if report.letter_file:
            logger.info(f"Report already has a letter file, deleting: {report.letter_file.name}")
            # Delete the existing file to avoid accumulating unused files
            try:
                report.letter_file.delete(save=False)
            except Exception as e:
                logger.warning(f"Could not delete existing letter file: {e}")

        # Save the new file
        report.letter_file.save(filename, ContentFile(buffer.getvalue()), save=False)
        report.letter_generated = True
        report.save()

        logger.info("Letter generated successfully")
        return report

    except Exception as e:
        import traceback
        logger.error(f"Error generating foreign donation letter: {str(e)}")
        logger.error(traceback.format_exc())

        # Add error info to report
        if hasattr(report, 'notes'):
            report.notes = (report.notes or '') + f"\nError generating letter: {str(e)}"
            report.save(update_fields=['notes'])

        raise