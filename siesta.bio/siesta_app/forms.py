from django import forms
from . import models

class FeatureExtractionForm(forms.ModelForm):
    class Meta:
        model = models.FeatureExtraction
        fields = '__all__'
        widgets = {
            'fs': forms.TextInput(attrs={'style': 'width: 100px; height: 30px;'}),
            'epoch': forms.TextInput(attrs={'style': 'width: 50px; height: 30px;'}),
            'ECoG1_chan': forms.TextInput(attrs={'style': 'width: 50px; height: 30px;'}),
            'ECoG2_chan': forms.TextInput(attrs={'style': 'width: 50px; height: 30px;'}),
            'EMG_chan': forms.TextInput(attrs={'style': 'width: 50px; height: 30px;'}),
        }