from django import forms

# Versi Ina
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
    

# Versi Bu Fitri
# class PerfumePreferenceForm(forms.Form):
#     # Choice options
#     SCENT_CHOICES = [
#         ('arabian', 'Arabian'),
#         ('aromatic', 'Aromatic'),
#         ('clean', 'Clean'),
#         ('citrus', 'Citrus'),
#         ('floral', 'Floral'),
#         ('fresh', 'Fresh'),
#         ('fruity', 'Fruity'),
#         ('jasmine', 'Jasmine'),
#         ('musk', 'Musk'),
#         ('oriental', 'Oriental'),
#         ('rose', 'Rose'),
#         ('sandalwood', 'Sandalwood'),
#         ('spicy', 'Spicy'),
#         ('sweet', 'Sweet'),
#         ('vanilla', 'Vanilla'),
#         ('woody', 'Woody'),
#     ]
    
#     CONCENTRATION_CHOICES = [
#         ('edp', 'Eau de Parfum (EDP)'),
#         ('edt', 'Eau de Toilette (EDT)'),
#     ]

#     DEPARTMENT_CHOICES = [
#         ('men', 'Men'),
#         ('women', 'Women'),
#         ('unisex', 'Unisex'),
#     ]

#     # Text fields
#     brand = forms.CharField(
#         max_length=100, 
#         label="Brand",
#         required=False,
#         widget=forms.TextInput(attrs={'placeholder': 'e.g. Chanel, Dior'})
#     )
    
#     name = forms.CharField(
#         max_length=100, 
#         label="Product Name",
#         required=False,
#         widget=forms.TextInput(attrs={'placeholder': 'e.g. Sauvage, Bleu'})
#     )
    
#     base_note = forms.CharField(
#         max_length=100, 
#         label="Base Note",
#         required=False,
#         widget=forms.TextInput(attrs={'placeholder': 'e.g. Vanilla, Sandalwood'})
#     )
    
#     middle_note = forms.CharField(
#         max_length=100, 
#         label="Middle Note",
#         required=False,
#         widget=forms.TextInput(attrs={'placeholder': 'e.g. Lavender, Jasmine'})
#     )
    
#     seller = forms.CharField(
#         max_length=100, 
#         label="Seller",
#         required=False,
#         widget=forms.TextInput(attrs={'placeholder': 'Seller name'})
#     )

#     # Choice fields
#     scent = forms.ChoiceField(
#         choices=SCENT_CHOICES, 
#         label="Scent",
#         widget=forms.Select(attrs={'class': 'form-select'})
#     )
    
#     department = forms.ChoiceField(
#         choices=DEPARTMENT_CHOICES, 
#         label="Department",
#         widget=forms.Select(attrs={'class': 'form-select'})
#     )
    
#     concentration = forms.ChoiceField(
#         choices=CONCENTRATION_CHOICES, 
#         label="Concentration",
#         widget=forms.Select(attrs={'class': 'form-select'})
#     )

#     # Numerical fields
#     ml = forms.FloatField(
#         label="Capacity (ml)",
#         min_value=0,
#         widget=forms.NumberInput(attrs={'step': '1'})
#     )
    
#     new_price = forms.FloatField(
#         label="Price ($)",
#         min_value=0,
#         widget=forms.NumberInput(attrs={'step': '0.01'})
#     )
    
#     item_rating = forms.FloatField(
#         label="Minimum Item Rating",
#         min_value=0,
#         max_value=5,
#         required=False,
#         widget=forms.NumberInput(attrs={'step': '0.1', 'placeholder': '0-5'})
#     )
    
#     seller_rating = forms.FloatField(
#         label="Minimum Seller Rating",
#         min_value=0,
#         max_value=5,
#         required=False,
#         widget=forms.NumberInput(attrs={'step': '0.1', 'placeholder': '0-5'})
#     )
    
#     num_seller_ratings = forms.IntegerField(
#         label="Minimum Seller Ratings Count",
#         min_value=0,
#         required=False,
#         widget=forms.NumberInput(attrs={'placeholder': 'e.g. 1000'})
#     )
    