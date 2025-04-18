from pathlib import Path
import os
from dotenv import load_dotenv
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-b_*n(q%r99nr9ta^0za2y_2tfld_arqti1i4)qup1bg2!5vn2f'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Load environment variables from .env file
load_dotenv()



INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'users',
    'api',
    'knox', 
    'django_rest_passwordreset',
    'chatbot',  # Chatbot app
    'finances',  # Financial management app
    'django_filters',  # For filtering in DRF
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ALLOWED_ORIGINS = [
 'http://localhost:5173',
]

AUTH_USER_MODEL = 'users.CustomUser'

AUTHENTICATION_BACKENDS = [
    # 'users.authback.EmailBackend',
    "django.contrib.auth.backends.ModelBackend", # this line fixed my problem
]

ROOT_URLCONF = 'auth.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR/"templates"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'auth.wsgi.application'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': ('knox.auth.TokenAuthentication',),
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend']
}

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myorg_db',  # Your database name
        'USER': 'postgres',  # Your PostgreSQL username (default is often 'postgres')
        'PASSWORD': '0000',  # Your PostgreSQL password
        'HOST': 'localhost',  # Or the host where your PostgreSQL server runs
        'PORT': '5432',  # Default PostgreSQL port
    }
}

MEDIA_URL = '/media/'  # The URL to access media files
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')  # The folder where media files will be stored
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760

# Create temp directory for file processing
TEMP_DIR = os.path.join(MEDIA_ROOT, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Create finance directories for upload organization
FINANCE_UPLOAD_DIR = os.path.join(MEDIA_ROOT, 'finance')
os.makedirs(os.path.join(FINANCE_UPLOAD_DIR, 'transactions'), exist_ok=True)
os.makedirs(os.path.join(FINANCE_UPLOAD_DIR, 'reports'), exist_ok=True)

# Chatbot settings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


#the email settings
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'noreply.myorg@gmail.com'
EMAIL_HOST_PASSWORD = 'ouht eeez vgiy xvbc'
DEFAULT_FROM_EMAIL = 'MyOrg'
# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'