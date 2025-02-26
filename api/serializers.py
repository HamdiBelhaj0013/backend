from rest_framework import serializers
from .models import *

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id','name','start_date','end_date','description','budget','status')
class MemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = ('id','name','address','email','nationality','birth_date','job','joining_date','role')
