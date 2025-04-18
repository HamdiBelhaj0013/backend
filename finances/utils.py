import os
import tempfile
from datetime import timedelta
from decimal import Decimal
from django.db.models import Sum, Q
from django.utils import timezone
import io
import csv


# This is intended for future PDF report generation
def generate_pdf_report(report_instance):
    """
    Generate a PDF report for the given financial report instance

    Note: This implementation requires 'reportlab' which should be added to requirements.txt
    To use this, uncomment the code below and install reportlab with:
    pip install reportlab
    """
    pass
    # Uncomment when ready to implement PDF reports
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    # Get transactions for the reporting period
    transactions = Transaction.objects.filter(
        date__gte=report_instance.start_date,
        date__lte=report_instance.end_date,
        status='verified'
    )

    # Set up the document
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Add title and summary
    elements.append(Paragraph(report_instance.title, title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        f"Period: {report_instance.start_date} to {report_instance.end_date}", 
        normal_style
    ))
    elements.append(Spacer(1, 12))

    # Calculate summary data
    total_income = transactions.filter(transaction_type='income').aggregate(
        total=Sum('amount')
    )['total'] or 0

    total_expenses = transactions.filter(transaction_type='expense').aggregate(
        total=Sum('amount')
    )['total'] or 0

    net_balance = total_income - total_expenses

    # Add financial summary
    elements.append(Paragraph("Financial Summary", heading_style))
    elements.append(Spacer(1, 6))

    summary_data = [
        ["Category", "Amount"],
        ["Total Income", f"{total_income:.2f}"],
        ["Total Expenses", f"{total_expenses:.2f}"],
        ["Net Balance", f"{net_balance:.2f}"]
    ]

    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (1, -1), 1, colors.black)
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Add transactions table
    elements.append(Paragraph("Transactions", heading_style))
    elements.append(Spacer(1, 6))

    # Prepare transaction data
    transaction_data = [
        ["Date", "Type", "Category", "Amount", "Description", "Project"]
    ]

    for t in transactions:
        transaction_data.append([
            t.date.strftime('%Y-%m-%d'),
            t.get_transaction_type_display(),
            t.get_category_display(),
            f"{t.amount:.2f}",
            t.description[:30] + '...' if len(t.description) > 30 else t.description,
            t.project.name if t.project else ''
        ])

    # Create transactions table
    if len(transaction_data) > 1:  # Only if we have transactions
        trans_table = Table(transaction_data)
        trans_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(trans_table)
    else:
        elements.append(Paragraph("No transactions for this period.", normal_style))

    # Build the PDF
    doc.build(elements)

    # Get the PDF content
    pdf_content = buffer.getvalue()
    buffer.close()

    # Save to report instance
    from django.core.files.base import ContentFile
    report_instance.report_file.save(
        f"{report_instance.title.replace(' ', '_')}.pdf",
        ContentFile(pdf_content)
    )

    return report_instance
    """


def get_financial_statistics(start_date=None, end_date=None):
    """
    Get financial statistics for a given period
    """
    from .models import Transaction, BudgetAllocation

    # Set default date range if not specified
    if not start_date:
        start_date = timezone.now().date() - timedelta(days=30)
    if not end_date:
        end_date = timezone.now().date()

    # Get verified transactions for the period
    transactions = Transaction.objects.filter(
        date__gte=start_date,
        date__lte=end_date,
        status='verified'
    )

    # Calculate totals
    total_income = transactions.filter(transaction_type='income').aggregate(
        total=Sum('amount')
    )['total'] or Decimal('0')

    total_expenses = transactions.filter(transaction_type='expense').aggregate(
        total=Sum('amount')
    )['total'] or Decimal('0')

    net_balance = total_income - total_expenses

    # Get income by category
    income_by_category = {}
    expense_by_category = {}

    for transaction_type, category_dict in [
        ('income', income_by_category),
        ('expense', expense_by_category)
    ]:
        categories = transactions.filter(
            transaction_type=transaction_type
        ).values('category').annotate(
            total=Sum('amount')
        )

        for category in categories:
            # Get display name for the category
            from .models import TRANSACTION_CATEGORIES
            category_display = dict(TRANSACTION_CATEGORIES).get(category['category'], category['category'])
            category_dict[category_display] = category['total']

    # Return statistics
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_balance': net_balance,
        'income_by_category': income_by_category,
        'expense_by_category': expense_by_category,
        'start_date': start_date,
        'end_date': end_date
    }


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