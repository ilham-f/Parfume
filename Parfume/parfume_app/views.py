import pandas as pd
import numpy as np
import re
import nltk
from django.shortcuts import redirect, render
from django.http import HttpResponse
from .forms import PerfumePreferenceForm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
import os

nltk.download('stopwords')
nltk.download('wordnet')

from django.conf import settings

def home(request):
    # Sample perfume data
    parfumes = [
        { "name": "Dior Sauvage", "price": 120 },
        { "name": "Chanel Bleu de Chanel", "price": 135 },
        { "name": "Yves Saint Laurent La Nuit De L'Homme", "price": 110 },
        { "name": "Creed Aventus", "price": 350 },
        { "name": "Tom Ford Oud Wood", "price": 250 },
        { "name": "Versace Eros", "price": 95 },
        { "name": "Armani Acqua di Gio", "price": 105 },
        { "name": "Jean Paul Gaultier Le Male", "price": 98 },
    ]

    # Get all previous preferences from cookies
    prev_parfume = {
        'scent': request.COOKIES.get('scent', '').capitalize(),
        'base_note': request.COOKIES.get('base_note', '').capitalize(),
        'middle_note': request.COOKIES.get('middle_note', '').capitalize(),
        'concentration': request.COOKIES.get('concentration', '').capitalize(),
        'ml': request.COOKIES.get('ml', ''),
        'new_price': request.COOKIES.get('price', ''),
    }

    # Get all input values from cookies
    scent = request.COOKIES.get('scent', '').lower()
    base_note = request.COOKIES.get('base_note', '').lower()
    middle_note = request.COOKIES.get('middle_note', '').lower()
    concentration = request.COOKIES.get('concentration', '').lower()
    ml = float(request.COOKIES.get('ml', 0))
    price = float(request.COOKIES.get('price', 0))

    # Load clustered perfume data
    csv_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'combinednotes_clustered.csv')
    data = pd.read_csv(csv_path, delimiter=';')
    
    data.dropna(subset=['scents', 'base_note', 'middle_note', 'combined_notes', 'ml', 'concentration', 'new_price'], inplace=True)

    # Load model TF-IDF dan KMeans
    vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'tfidf_vectorizer.pkl'))
    kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'kmeans_model.pkl'))

    # Gabungkan notes untuk vektorisasi
    combined_input = f"{scent} {base_note} {middle_note}".strip()
    input_vector_text = vectorizer.transform([combined_input])

    # Fitur numerik
    input_note_length = len(combined_input)
    input_unique_words = len(set(combined_input.split()))
    input_numerical = csr_matrix([[input_note_length, input_unique_words]])

    # Gabungkan semua fitur
    input_vector = hstack([input_vector_text, input_numerical])

    # Prediksi cluster
    input_cluster = kmeans.predict(input_vector)[0]
    cluster_parfumes = data[data['cluster'] == input_cluster].copy()

    # Hitung cosine similarity
    cluster_vectors = vectorizer.transform(cluster_parfumes['combined_notes'])
    cluster_numerical = csr_matrix(cluster_parfumes[['note_length', 'unique_words']].values)
    cluster_combined = hstack([cluster_vectors, cluster_numerical])
    similarity_scores = cosine_similarity(input_vector, cluster_combined)
    cluster_parfumes['similarity_score'] = similarity_scores[0]

    # Filter preferensi user
    filtered = cluster_parfumes[
        (cluster_parfumes['ml'] >= ml - 10) &
        (cluster_parfumes['ml'] <= ml + 10) &
        (cluster_parfumes['concentration'].str.contains(concentration, case=False)) &
        (cluster_parfumes['new_price'] <= price)
    ]

    # Ambil 5 parfum teratas
    recommended = filtered.sort_values('similarity_score', ascending=False).head(8)
    recommended['similarity_percentage'] = (recommended['similarity_score'] * 100).round(2)

    # Prepare context
    context = {
        "parfumes": parfumes,
        "prev_parfume": prev_parfume,
        "rec_parfumes": recommended[[
            'brand', 'name', 'scents', 'base_note', 'middle_note',
            'ml', 'concentration', 'new_price', 'similarity_percentage'
        ]].to_dict(orient='records')
    }
    
    return render(request, 'home.html', context)

# def home(request):
#     # Sample perfume data
#     parfumes = [
#         { "name": "Dior Sauvage", "price": 120 },
#         { "name": "Chanel Bleu de Chanel", "price": 135 },
#         { "name": "Yves Saint Laurent La Nuit De L'Homme", "price": 110 },
#         { "name": "Creed Aventus", "price": 350 },
#         { "name": "Tom Ford Oud Wood", "price": 250 },
#         { "name": "Versace Eros", "price": 95 },
#         { "name": "Armani Acqua di Gio", "price": 105 },
#         { "name": "Jean Paul Gaultier Le Male", "price": 98 },
#     ]

#     # Get all previous preferences from cookies
#     prev_parfume = {
#         'scent': request.COOKIES.get('scent', '').capitalize(),
#         'base_note': request.COOKIES.get('base_note', '').capitalize(),
#         'middle_note': request.COOKIES.get('middle_note', '').capitalize(),
#         'brand': request.COOKIES.get('brand', '').capitalize(),
#         'name': request.COOKIES.get('name', '').capitalize(),
#         'department': request.COOKIES.get('department', '').capitalize(),
#         'concentration': request.COOKIES.get('concentration', '').capitalize(),
#         'seller': request.COOKIES.get('seller', '').capitalize(),
#         'ml': request.COOKIES.get('ml', ''),
#         'new_price': request.COOKIES.get('new_price', ''),
#         'item_rating': request.COOKIES.get('item_rating', ''),
#         'seller_rating': request.COOKIES.get('seller_rating', ''),
#         'num_seller_ratings': request.COOKIES.get('num_seller_ratings', '')
#     }

#     # Get all input values from cookies
#     scent = request.COOKIES.get('scent', '').lower()
#     base_note = request.COOKIES.get('base_note', '').lower()
#     middle_note = request.COOKIES.get('middle_note', '').lower()
#     brand = request.COOKIES.get('brand', '').lower()
#     name = request.COOKIES.get('name', '').lower()
#     department = request.COOKIES.get('department', '').lower()
#     concentration = request.COOKIES.get('concentration', '').lower()
#     seller = request.COOKIES.get('seller', '').lower()
#     ml = float(request.COOKIES.get('ml', 0))
#     new_price = float(request.COOKIES.get('new_price', 0))
#     item_rating = float(request.COOKIES.get('item_rating', 0))
#     seller_rating = float(request.COOKIES.get('seller_rating', 0))
#     num_seller_ratings = float(request.COOKIES.get('num_seller_ratings', 0))

#     # Load clustered perfume data
#     csv_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'clustered_parfume.csv')
#     data = pd.read_csv(csv_path, delimiter=';')
    
#     # Drop rows with missing values in important columns
#     columns_to_check = ['scents', 'base_note', 'middle_note', 'brand', 'name', 
#                       'department', 'concentration', 'seller', 'new_price', 'ml',
#                       'item_rating', 'seller_rating', 'num_seller_ratings']
#     data.dropna(subset=columns_to_check, inplace=True)

#     # Load models
#     vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'tfidf_vectorizer_all.pkl'))
#     kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'kmeans_model_all.pkl'))
#     scaler = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'scaler.pkl'))

#     # Combine text features for vectorization
#     combined_input = f"{scent} {base_note} {middle_note} {brand} {name} {department} {concentration} {seller}".strip()
#     input_vector_text = vectorizer.transform([combined_input])

#     # Prepare numerical features
#     numerical_input = [[new_price, ml, item_rating, seller_rating, num_seller_ratings]]
#     scaled_numerical = scaler.transform(numerical_input)
#     input_numerical = csr_matrix(scaled_numerical)

#     # Combine all features
#     input_vector = hstack([input_vector_text, input_numerical])

#     # Predict cluster
#     input_cluster = kmeans.predict(input_vector)[0]
#     cluster_parfumes = data[data['cluster'] == input_cluster].copy()

#     # Calculate similarity
#     cluster_text = vectorizer.transform(
#         cluster_parfumes['scents'] + ' ' + 
#         cluster_parfumes['base_note'] + ' ' + 
#         cluster_parfumes['middle_note'] + ' ' +
#         cluster_parfumes['brand'] + ' ' +
#         cluster_parfumes['name'] + ' ' +
#         cluster_parfumes['department'] + ' ' +
#         cluster_parfumes['concentration'] + ' ' +
#         cluster_parfumes['seller']
#     )
    
#     cluster_numerical = csr_matrix(scaler.transform(
#         cluster_parfumes[['new_price', 'ml', 'item_rating', 'seller_rating', 'num_seller_ratings']]
#     ))
    
#     cluster_combined = hstack([cluster_text, cluster_numerical])
#     similarity_scores = cosine_similarity(input_vector, cluster_combined)
#     cluster_parfumes['similarity_score'] = similarity_scores[0]

#     # Apply user filters
#     filtered = cluster_parfumes[
#         (cluster_parfumes['ml'] >= ml - 10) &
#         (cluster_parfumes['ml'] <= ml + 10) &
#         (cluster_parfumes['concentration'].str.contains(concentration, case=False)) &
#         (cluster_parfumes['new_price'] <= new_price) &
#         (cluster_parfumes['item_rating'] >= item_rating) &
#         (cluster_parfumes['seller_rating'] >= seller_rating)
#     ]

#     # Get top 8 recommendations
#     recommended = filtered.sort_values('similarity_score', ascending=False).head(8)
#     recommended['similarity_percentage'] = (recommended['similarity_score'] * 100).round(2)

#     # Prepare context
#     context = {
#         "parfumes": parfumes,
#         "prev_parfume": prev_parfume,
#         "rec_parfumes": recommended[[
#             'brand', 'name', 'scents', 'base_note', 'middle_note',
#             'ml', 'concentration', 'new_price', 'item_rating',
#             'seller_rating', 'num_seller_ratings', 'similarity_percentage'
#         ]].to_dict(orient='records')
#     }
    
#     return render(request, 'home.html', context)

def about(request):
    return render(request, 'about.html')

# Versi Ina
def recommendations(request):
    if request.method == "POST":
        form = PerfumePreferenceForm(request.POST)
        
        def clean_word_input(text):
            # Hapus karakter non-alfabet dan non-spasi
            if text:
                # Hanya ambil huruf dan spasi
                cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
                cleaned = cleaned.strip()
                return cleaned if cleaned else ''
            return ''

        if form.is_valid():
            scent = clean_word_input(form.cleaned_data.get("scent", "").lower())
            base_note = clean_word_input(form.cleaned_data.get("base_note", "").lower())
            middle_note = clean_word_input(form.cleaned_data.get("middle_note", "").lower())
            ml = form.cleaned_data.get("ml") or 0
            concentration = clean_word_input(form.cleaned_data.get("concentration", "").lower())
            price = form.cleaned_data.get("price") or 0
            
            print("base_note:", repr(base_note))
            print("middle_note:", repr(middle_note))
            
            if not any([base_note, middle_note]):
                context = {"parfumes": []}
                response = render(request, 'result.html', context)
                # Set cookie seperti biasa
                response.set_cookie('scent', scent)
                response.set_cookie('base_note', base_note)
                response.set_cookie('middle_note', middle_note)
                response.set_cookie('ml', ml)
                response.set_cookie('concentration', concentration)
                response.set_cookie('price', price)
                return response

            # Load dataset parfum yang sudah memiliki hasil clustering
            csv_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'combinednotes_clustered.csv')
            data = pd.read_csv(csv_path, delimiter=';')
            data.dropna(subset=['scents', 'base_note', 'middle_note', 'combined_notes', 'ml', 'concentration', 'new_price'], inplace=True)

            # Load model TF-IDF dan KMeans
            vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'tfidf_vectorizer.pkl'))
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'kmeans_model.pkl'))

            # Gabungkan notes untuk vektorisasi
            combined_input = f"{scent} {base_note} {middle_note}".strip()
            input_vector_text = vectorizer.transform([combined_input])

            # Fitur numerik
            input_note_length = len(combined_input)
            input_unique_words = len(set(combined_input.split()))
            input_numerical = csr_matrix([[input_note_length, input_unique_words]])

            # Gabungkan semua fitur
            input_vector = hstack([input_vector_text, input_numerical])
            print(input_vector)

            # Prediksi cluster
            input_cluster = kmeans.predict(input_vector)[0]
            cluster_parfumes = data[data['cluster'] == input_cluster].copy()

            # Hitung cosine similarity
            cluster_vectors = vectorizer.transform(cluster_parfumes['combined_notes'])
            cluster_numerical = csr_matrix(cluster_parfumes[['note_length', 'unique_words']].values)
            cluster_combined = hstack([cluster_vectors, cluster_numerical])
            similarity_scores = cosine_similarity(input_vector, cluster_combined)
            cluster_parfumes['similarity_score'] = similarity_scores[0]

            # Filter preferensi user
            filtered = cluster_parfumes[
                (cluster_parfumes['ml'] >= ml - 10) &
                (cluster_parfumes['ml'] <= ml + 10) &
                (cluster_parfumes['concentration'].str.contains(concentration, case=False)) &
                (cluster_parfumes['new_price'] <= price)
            ]

            # Ambil 5 parfum teratas
            recommended = filtered.sort_values('similarity_score', ascending=False).head(5)
            # --- PERBAIKAN DI SINI ---
            # Untuk mengambil subset vparfum (matriks fitur) untuk data rekomendasi tersebut:
            # Kita perlu mendapatkan indeks posisi dari 'recommended' dalam 'cluster_parfumes'
            # karena cluster_combined adalah matriks sparse yang diindeks berdasarkan posisi (iloc).
            
            # Dapatkan indeks asli dari DataFrame 'recommended'
            recommended_original_indices = recommended.index.tolist()

            # Dapatkan indeks posisi (iloc) dari item-item 'recommended' dalam 'cluster_parfumes'
            # Ini mengasumsikan cluster_parfumes.index berisi semua indeks dari cluster asli.
            positional_indices_in_cluster_combined = cluster_parfumes.index.get_indexer(recommended_original_indices)
            
            # Ambil subset matriks fitur (vparfum_top5) menggunakan indeks posisi
            vparfum_top5 = cluster_combined[positional_indices_in_cluster_combined]
            print("\nMatriks Fitur (vparfum_top5) dari 5 Parfum Teratas yang Direkomendasikan:")
            print(vparfum_top5)
            # --- AKHIR PERBAIKAN ---
            
            recommended['similarity_percentage'] = (recommended['similarity_score'] * 100).round(2)

            # Siapkan context
            context = {
                "parfumes": recommended[['brand', 'name', 'scents', 'base_note', 'middle_note',
                                         'ml', 'concentration', 'new_price', 'similarity_percentage']].to_dict(orient='records')
            }

            # Render response dan set cookie
            response = render(request, 'result.html', context)
            response.set_cookie('scent', scent)
            response.set_cookie('base_note', base_note)
            response.set_cookie('middle_note', middle_note)
            response.set_cookie('ml', ml)
            response.set_cookie('concentration', concentration)
            response.set_cookie('price', price)

            return response
    else:
        # Ambil preferensi terakhir dari cookie saat GET
        form = PerfumePreferenceForm(initial={
            'scent': request.COOKIES.get('scent', ''),
            'base_note': request.COOKIES.get('base_note', ''),
            'middle_note': request.COOKIES.get('middle_note', ''),
            'ml': request.COOKIES.get('ml', ''),
            'concentration': request.COOKIES.get('concentration', ''),
            'price': request.COOKIES.get('price', ''),
        })

    # Tampilkan form
    return render(request, 'recommendation_form.html', {'form': form})

# Versi Bu Fitri
# def recommendations(request):
#     if request.method == "POST":
#         form = PerfumePreferenceForm(request.POST)

#         if form.is_valid():
#             # Get all input from form except old_price
#             scent = form.cleaned_data.get("scent", "").lower()
#             base_note = form.cleaned_data.get("base_note", "").lower()
#             middle_note = form.cleaned_data.get("middle_note", "").lower()
#             brand = form.cleaned_data.get("brand", "").lower()
#             name = form.cleaned_data.get("name", "").lower()
#             department = form.cleaned_data.get("department", "").lower()
#             concentration = form.cleaned_data.get("concentration", "").lower()
#             seller = form.cleaned_data.get("seller", "").lower()
#             ml = form.cleaned_data.get("ml") or 0
#             new_price = form.cleaned_data.get("new_price") or 0
#             item_rating = form.cleaned_data.get("item_rating") or 0
#             seller_rating = form.cleaned_data.get("seller_rating") or 0
#             num_seller_ratings = form.cleaned_data.get("num_seller_ratings") or 0

#             # Load clustered perfume data
#             csv_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'clustered_parfume.csv')
#             data = pd.read_csv(csv_path, delimiter=';')
            
#             # Drop rows with missing values in important columns
#             columns_to_check = ['scents', 'base_note', 'middle_note', 'brand', 'name', 
#                               'department', 'concentration', 'seller', 'new_price', 'ml',
#                               'item_rating', 'seller_rating', 'num_seller_ratings']
#             data.dropna(subset=columns_to_check, inplace=True)

#             # Load models
#             vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'tfidf_vectorizer_all.pkl'))
#             kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'kmeans_model_all.pkl'))
#             scaler = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'model', 'scaler.pkl'))

#             # Combine text features
#             combined_input = f"{scent} {base_note} {middle_note} {brand} {name} {department} {concentration} {seller}".strip()
#             input_vector_text = vectorizer.transform([combined_input])

#             # Prepare numerical features
#             numerical_input = [[new_price, ml, item_rating, seller_rating, num_seller_ratings]]
#             scaled_numerical = scaler.transform(numerical_input)
#             input_numerical = csr_matrix(scaled_numerical)

#             # Combine all features
#             input_vector = hstack([input_vector_text, input_numerical])

#             # Predict cluster
#             input_cluster = kmeans.predict(input_vector)[0]
#             cluster_parfumes = data[data['cluster'] == input_cluster].copy()

#             # Calculate similarity
#             cluster_text = vectorizer.transform(
#                 cluster_parfumes['scents'] + ' ' + 
#                 cluster_parfumes['base_note'] + ' ' + 
#                 cluster_parfumes['middle_note'] + ' ' +
#                 cluster_parfumes['brand'] + ' ' +
#                 cluster_parfumes['name'] + ' ' +
#                 cluster_parfumes['department'] + ' ' +
#                 cluster_parfumes['concentration'] + ' ' +
#                 cluster_parfumes['seller']
#             )
            
#             cluster_numerical = csr_matrix(scaler.transform(
#                 cluster_parfumes[['new_price', 'ml', 'item_rating', 'seller_rating', 'num_seller_ratings']]
#             ))
            
#             cluster_combined = hstack([cluster_text, cluster_numerical])
#             similarity_scores = cosine_similarity(input_vector, cluster_combined)
#             cluster_parfumes['similarity_score'] = similarity_scores[0]

#             # Apply user filters
#             filtered = cluster_parfumes[
#                 (cluster_parfumes['ml'] >= ml - 10) &
#                 (cluster_parfumes['ml'] <= ml + 10) &
#                 (cluster_parfumes['concentration'].str.contains(concentration, case=False)) &
#                 (cluster_parfumes['new_price'] <= new_price) 
#                 # (cluster_parfumes['item_rating'] >= item_rating) &
#                 # (cluster_parfumes['seller_rating'] >= seller_rating) &
#                 # (cluster_parfumes['num_seller_ratings'] >= num_seller_ratings)
#             ]

#             # Get top 5 recommendations
#             recommended = filtered.sort_values('similarity_score', ascending=False).head(5)
#             recommended['similarity_percentage'] = (recommended['similarity_score'] * 100).round(2)

#             # Prepare context
#             context = {
#                 "parfumes": recommended[[
#                     'brand', 'name', 'scents', 'base_note', 'middle_note',
#                     'ml', 'concentration', 'new_price', 'item_rating',
#                     'seller_rating', 'num_seller_ratings', 'similarity_percentage'
#                 ]].to_dict(orient='records')
#             }

#             # Set cookies for all inputs
#             response = render(request, 'result.html', context)
#             for field in form.cleaned_data:
#                 if field != 'old_price':  # Skip old_price
#                     response.set_cookie(field, form.cleaned_data[field])
            
#             return response
#     else:
#         # Initialize form with cookie values
#         initial_data = {field: request.COOKIES.get(field, '') for field in [
#             'scent', 'base_note', 'middle_note', 'brand', 'name',
#             'department', 'concentration', 'seller', 'ml', 'new_price',
#             'item_rating', 'seller_rating', 'num_seller_ratings'
#         ]}
#         form = PerfumePreferenceForm(initial=initial_data)

#     return render(request, 'recommendation_form.html', {'form': form})

# Versi Ina
def recommendation_form(request):
    form = PerfumePreferenceForm(initial={
        'scent': request.COOKIES.get('scent', ''),
        'base_note': request.COOKIES.get('base_note', ''),
        'middle_note': request.COOKIES.get('middle_note', ''),
        'ml': request.COOKIES.get('ml', ''),
        'concentration': request.COOKIES.get('concentration', ''),
        'price': request.COOKIES.get('price', ''),
    })
    
    return render(request, 'recommendation_form.html', {'form': form})

# Versi Bu Fitri
# def recommendation_form(request):
#     form = PerfumePreferenceForm(initial={
#         'scent': request.COOKIES.get('scent', ''),
#         'base_note': request.COOKIES.get('base_note', ''),
#         'middle_note': request.COOKIES.get('middle_note', ''),
#         'brand': request.COOKIES.get('brand', ''),
#         'name': request.COOKIES.get('name', ''),
#         'department': request.COOKIES.get('department', ''),
#         'concentration': request.COOKIES.get('concentration', ''),
#         'seller': request.COOKIES.get('seller', ''),
#         'ml': request.COOKIES.get('ml', ''),
#         'new_price': request.COOKIES.get('new_price', ''),
#         'item_rating': request.COOKIES.get('item_rating', ''),
#         'seller_rating': request.COOKIES.get('seller_rating', ''),
#         'num_seller_ratings': request.COOKIES.get('num_seller_ratings', '')
#     })
    
#     return render(request, 'recommendation_form.html', {'form': form}