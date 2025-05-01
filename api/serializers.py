from .models import *
from rest_framework import serializers


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id', 'name', 'start_date', 'end_date', 'description', 'budget', 'status')


class MemberSerializer(serializers.ModelSerializer):
    association = serializers.PrimaryKeyRelatedField(
        queryset=AssociationAccount.objects.all(),
        required=False
    )
    association_name = serializers.SerializerMethodField()

    class Meta:
        model = Member
        fields = [
            'id', 'name', 'cin', 'email', 'birth_date', 'address',
            'nationality', 'job', 'joining_date', 'role',
            'association', 'association_name', 'needs_profile_completion'
        ]

    def get_association_name(self, obj):
        return obj.association.name if obj.association else None

    def validate(self, data):
        """Validate that all required fields have proper values"""
        # If this is an update operation (existing instance)
        if self.instance:
            # Combine existing values with updates
            name = data.get('name', self.instance.name)
            cin = data.get('cin', self.instance.cin)
            birth_date = data.get('birth_date', self.instance.birth_date)
            address = data.get('address', self.instance.address)
            nationality = data.get('nationality', self.instance.nationality)
            job = data.get('job', self.instance.job)

            # Check if any field is missing or has a default placeholder value
            needs_completion = (
                    not name or
                    not cin or
                    not birth_date or
                    not address or address == "Please update your address" or
                    not nationality or nationality == "Please update your nationality" or
                    not job or job == "Please update your job"
            )

            # Only update the needs_profile_completion flag if all fields are properly filled
            data['needs_profile_completion'] = needs_completion

            # Log what's happening for debugging
            print(f"Member update validation: needs_profile_completion = {needs_completion}")
            if needs_completion:
                print(f"Fields still needed: " +
                      (f"name ('{name}')" if not name else "") +
                      (f"cin ('{cin}')" if not cin else "") +
                      (f"birth_date ('{birth_date}')" if not birth_date else "") +
                      (f"address ('{address}')" if not address or address == "Please update your address" else "") +
                      (
                          f"nationality ('{nationality}')" if not nationality or nationality == "Please update your nationality" else "") +
                      (f"job ('{job}')" if not job or job == "Please update your job" else ""))

        return data