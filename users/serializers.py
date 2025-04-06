from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import CustomUser, AssociationAccount


class AssociationSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssociationAccount
        fields = ['id', 'name', 'email', 'matricule_fiscal']


class UserProfileSerializer(serializers.ModelSerializer):
    association = AssociationSerializer(read_only=True)

    class Meta:
        model = CustomUser
        fields = [
            'id',
            'email',
            'full_name',
            'is_staff',
            'is_superuser',
            'association'
        ]


User = get_user_model()


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


# Association Account Registration Serializer
class AssociationRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssociationAccount
        fields = ('id', 'name', 'email', 'cin_recto', 'cin_verso', 'matricule_fiscal', 'rne_document')

    def create(self, validated_data):
        return super().create(validated_data)


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