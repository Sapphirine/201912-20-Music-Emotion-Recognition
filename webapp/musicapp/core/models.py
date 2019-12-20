from django.db import models

# Create your models here.
class Music(models.Model):
    name = models.CharField(max_length=100)
    music = models.FileField(upload_to='music/clips/')
   
    def __str__(self):
        return self.name