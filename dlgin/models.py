from django.db import models


class Upload(models.Model):
    document = models.FileField(upload_to='songs/')
