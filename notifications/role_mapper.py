ROLE_MAPPING = {

    'admin': 'president',
    'president': 'president',
    'treasurer': 'treasurer',
    'secretary': 'secretary',
    'member': 'member',
}

REVERSE_MAPPING = {
    'president': 'admin',
    'treasurer': 'treasurer',
    'secretary': 'secretary',
    'member': 'member',
}
def get_frontend_role(backend_role):
    return ROLE_MAPPING.get(backend_role, backend_role)


def get_backend_role(frontend_role):
    return REVERSE_MAPPING.get(frontend_role, frontend_role)