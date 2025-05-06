# save as env_check.py
import sys
import os
import pkg_resources


def check_environment():
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)

    print("\nInstalled packages:")
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    for pkg in installed_packages_list:
        print(pkg)

    print("\nVirtual environment info:")
    print("Virtual env:", os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment'))

    # Check for potential conflicts
    potential_conflicts = ["numpy", "torch", "tensorflow", "typing-extensions"]
    print("\nPotential conflict packages:")
    for package in potential_conflicts:
        try:
            pkg = pkg_resources.get_distribution(package)
            print(f"{package}: {pkg.version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Not installed")


if __name__ == "__main__":
    check_environment()