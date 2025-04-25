from .models import *
from rest_framework import serializers


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id', 'name', 'start_date', 'end_date', 'description', 'budget', 'status')


from .models import *
from rest_framework import serializers


class MemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = '__all__'
        # Make association field read-only to prevent the validation error
        read_only_fields = ('association',)

    def validate(self, data):
        """
        Custom validation for the Member model
        """
        # Log the data being validated for debugging
        print("Validating Member data:", data)

        # Make sure required fields are present
        required_fields = ['name', 'address', 'email', 'nationality', 'birth_date', 'job', 'joining_date', 'role']
        for field in required_fields:
            if field not in data:
                raise serializers.ValidationError(f"{field} is required")

        # Ensure needs_profile_completion has a value if not provided
        if 'needs_profile_completion' not in data:
            data['needs_profile_completion'] = False

        return data