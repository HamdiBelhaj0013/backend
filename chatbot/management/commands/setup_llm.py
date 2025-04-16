import os
import sys
import requests
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Download and set up the LLM for the chatbot'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            help='Model filename to download'
        )

        parser.add_argument(
            '--url',
            type=str,
            default='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            help='URL to download the model from'
        )

    def handle(self, *args, **options):
        model_name = options['model']
        model_url = options['url']

        # Create the model directory if it doesn't exist
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, model_name)

        # Check if the model already exists
        if os.path.exists(model_path):
            self.stdout.write(self.style.SUCCESS(f'Model already exists at {model_path}'))
            return

        self.stdout.write(f'Downloading model from {model_url}...')
        self.stdout.write(f'This may take a while depending on your connection speed.')

        # Download the model with progress indication
        with open(model_path, 'wb') as f:
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            if total_size == 0:
                self.stdout.write(self.style.WARNING('Unknown file size, downloading without progress indication'))
                f.write(response.content)
            else:
                downloaded = 0
                total_size_mb = total_size / (1024 * 1024)

                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)

                    # Print progress
                    done = int(50 * downloaded / total_size)
                    downloaded_mb = downloaded / (1024 * 1024)
                    sys.stdout.write(
                        f'\r[{"=" * done}{" " * (50 - done)}] {downloaded_mb:.1f}/{total_size_mb:.1f} MB ({100 * downloaded / total_size:.1f}%)')
                    sys.stdout.flush()

                sys.stdout.write('\n')

        self.stdout.write(self.style.SUCCESS(f'Successfully downloaded model to {model_path}'))