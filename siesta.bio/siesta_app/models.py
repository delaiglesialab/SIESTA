from django.db import models
from django.core.validators import MinValueValidator


class FeatureExtraction(models.Model):
    fs = models.IntegerField(verbose_name="Sampling Rate", default=400, validators=[MinValueValidator(1)])
    epoch = models.IntegerField(verbose_name="Epoch", default=10, validators=[MinValueValidator(1)])
    ECoG1_chan = models.IntegerField(verbose_name="ECoG1 Channel", default=1, validators=[MinValueValidator(1)])
    ECoG2_chan = models.IntegerField(verbose_name="ECoG2 Channel", blank=True, null=True, validators=[MinValueValidator(1)])
    EMG_chan = models.IntegerField(verbose_name="EMG Channel", default=3, validators=[MinValueValidator(1)])