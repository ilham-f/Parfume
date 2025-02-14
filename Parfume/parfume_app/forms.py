from django import forms

class PerfumePreferenceForm(forms.Form):
    SCENT_CHOICES = [
        ('arabian', 'Arabian'),
        ('aromatic', 'Aromatic'),
        ('clean', 'Clean'),
        ('citrus', 'Citrus'),
        ('floral', 'Floral'),
        ('fresh', 'Fresh'),
        ('fruity', 'Fruity'),
        ('jasmine', 'Jasmine'),
        ('musk', 'Musk'),
        ('oriental', 'Oriental'),
        ('rose', 'Rose'),
        ('sandalwood', 'Sandalwood'),
        ('spicy', 'Spicy'),
        ('sweet', 'Sweet'),
        ('vanilla', 'Vanilla'),
        ('woody', 'Woody'),
    ]
    
    CONCENTRATION_CHOICES = [
        ('edp', 'Eau de Parfum (EDP)'),
        ('edt', 'Eau de Toilette (EDT)'),
    ]

    scent = forms.ChoiceField(choices=SCENT_CHOICES, label="Scent")
    base_note = forms.CharField(max_length=100, label="Base Note")
    middle_note = forms.CharField(max_length=100, label="Middle Note")
    ml = forms.FloatField(label="Capacity (ml)")
    concentration = forms.ChoiceField(choices=CONCENTRATION_CHOICES, label="Concentration")
    price = forms.FloatField(label="Maximum Price (in dollars)")