import os
from django.core.management.base import BaseCommand
from django.conf import settings
from chatbot.utils import process_policy_document


class Command(BaseCommand):
    help = 'Initialize the chatbot with the policy document'

    def add_arguments(self, parser):
        parser.add_argument('--pdf_path', type=str, help='Path to the policy document PDF')

    def handle(self, *args, **options):
        pdf_path = options.get('pdf_path')

        if not pdf_path:
            self.stdout.write(self.style.ERROR('No PDF path provided. Please specify --pdf_path.'))
            return

        if not os.path.exists(pdf_path):
            self.stdout.write(self.style.ERROR(f'PDF file not found at {pdf_path}'))
            return

        try:
            document = process_policy_document(pdf_path)
            self.stdout.write(self.style.SUCCESS(f'Successfully processed document: {document.title}'))
            self.stdout.write(self.style.SUCCESS(f'Created {document.chunks.count()} document chunks with embeddings'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error processing document: {e}'))