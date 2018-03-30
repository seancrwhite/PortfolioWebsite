from django import forms
from dlgin.models import Upload


class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('document',)

    def pack(self):
        """
        Packs valid song uploads into lists and sends to Model for processing
        :return:
        """
        slices = []

        for form in self.forms:
            break  # TODO: implement file upload