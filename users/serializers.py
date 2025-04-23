from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import CustomUser, AssociationAccount, Role
from .permissions import ROLE_PERMISSIONS


User = get_user_model()


class AssociationAccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssociationAccount
        fields = [
            'id', 'name', 'email',
            'president_email', 'president_name',
            'treasurer_email', 'treasurer_name',
            'secretary_email', 'secretary_name',
            'cin_recto', 'cin_verso', 'matricule_fiscal', 'rne_document', 'is_verified',
            'verification_date', 'verification_status', 'verification_notes'
        ]
        read_only_fields = [
            'is_verified', 'verification_date', 'verification_status', 'verification_notes'
        ]


# Association Account Registration Serializer
class AssociationRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssociationAccount
        fields = ('id', 'name', 'email',
                'president_email', 'president_name',
                'treasurer_email', 'treasurer_name',
                'secretary_email', 'secretary_name',
                'cin_recto', 'cin_verso', 'matricule_fiscal', 'rne_document')

    def create(self, validated_data):
        return super().create(validated_data)


class AssociationVerificationSerializer(serializers.ModelSerializer):
    """Serializer for verifying an association account"""
    class Meta:
        model = AssociationAccount
        fields = [
            'id', 'verification_status', 'verification_notes',
            'is_verified', 'verification_date'
        ]
        read_only_fields = fields


class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = ['id', 'name', 'description']


class UserProfileSerializer(serializers.ModelSerializer):
    association = AssociationAccountSerializer(read_only=True)
    role = RoleSerializer(read_only=True)
    role_id = serializers.PrimaryKeyRelatedField(
        queryset=Role.objects.all(),
        source='role',
        write_only=True,
        required=False
    )
    permissions = serializers.SerializerMethodField()

    class Meta:
        model = CustomUser
        fields = [
            'id',
            'email',
            'full_name',
            'is_staff',
            'is_superuser',
            'association',
            'role',
            'role_id',
            'permissions'
        ]

    def get_permissions(self, obj):
        """Return all permissions for this user's role"""
        if not obj.role:
            return {}

        return ROLE_PERMISSIONS.get(obj.role.name, {})


# Login Serializer
class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)  # Hide password in response

    def to_representation(self, instance):
        """ Modify response to include association info and full_name """
        ret = super().to_representation(instance)
        ret.pop('password', None)

        # Add full_name to the response
        ret['full_name'] = instance.full_name if hasattr(instance, 'full_name') else None

        if hasattr(instance, 'association') and instance.association:
            ret['association'] = {
                "id": instance.association.id,
                "name": instance.association.name
            }

        return ret


# User Registration Serializer
class RegisterSerializer(serializers.ModelSerializer):
    association_id = serializers.PrimaryKeyRelatedField(
        queryset=AssociationAccount.objects.all(),
        source="association",
        write_only=True
    )
    association = serializers.SerializerMethodField()
    full_name = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ('id', 'email', 'password', 'full_name', 'association_id', 'association')
        extra_kwargs = {'password': {'write_only': True}}

    def get_association(self, obj):
        """ Return association details in response """
        if obj.association:
            return {"id": obj.association.id, "name": obj.association.name}
        return None

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)  # Ensure password is hashed
        user.save()
        return user