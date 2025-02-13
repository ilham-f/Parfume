from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from .forms import PerfumePreferenceForm
from django.db.models import Count, Avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def home(request):
    parfumes = [
        { "name": "Dior Sauvage", "price": 120000 },
        { "name": "Chanel Bleu de Chanel", "price": 135000 },
        { "name": "Yves Saint Laurent La Nuit De L'Homme", "price": 110000 },
        { "name": "Creed Aventus", "price": 350000 },
        { "name": "Tom Ford Oud Wood", "price": 250000 },
        { "name": "Versace Eros", "price": 95000 },
        { "name": "Armani Acqua di Gio", "price": 105000 },
        { "name": "Paco Rabanne 1 Million", "price": 100000 },
        { "name": "Jean Paul Gaultier Le Male", "price": 98000 },
        { "name": "Byredo Gypsy Water", "price": 180000 }
    ]

    context = {
        "parfumes": parfumes
    }
    return render(request, 'home.html', context)

def recommendation(request):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    # Load CSV
    file_csv = 'data_parfume.csv'  # Replace with the path to your CSV file
    data = pd.read_csv(file_csv, delimiter=';')

    # Process Cleaning Data
    data = data.drop_duplicates()
    data = data.dropna(subset=['scents', 'base_note', 'middle_note', 'name', 'brand', 'new_price', 'ml', 'concentration'])
    data['new_price'] = pd.to_numeric(data['new_price'], errors='coerce')
    data['ml'] = pd.to_numeric(data['ml'], errors='coerce')
    data = data.dropna(subset=['new_price', 'ml'])

    # Menghapus karakter non-alfabet dan ubah ke huruf kecil
    data['scents'] = data['scents'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    data['base_note'] = data['base_note'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    data['middle_note'] = data['middle_note'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    data['concentration'] = data['concentration'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

    # Kombinasi 'scents', 'base_note', and 'middle_note' menjadi satu string
    data['combined_notes'] = data['scents'] + ' ' + data['base_note'] + ' ' + data['middle_note']

    # menggunakan TF-IDF Vectorizer untuk mengubah teks menjadi representasi numerik
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['combined_notes'])

    # Elbow Method untuk menentukan jumlah cluster optimal
    inertia = []
    K = range(1, 11)  # Coba jumlah cluster dari 1 hingga 10
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)  # Simpan inertia untuk setiap k

    # Pilih jumlah cluster berdasarkan Elbow Method
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)  # Tambahkan kolom 'cluster' ke DataFrame
    data.to_csv('after-cluster.csv', index=False)

    # Simpan tiap cluster ke dalam file CSV terpisah
    for cluster_id in range(n_clusters):
        # Ambil data parfum dalam cluster tertentu
        cluster_data = data[data['cluster'] == cluster_id]
        
        # Simpan ke file CSV
        cluster_data.to_csv(f'cluster_{cluster_id}.csv', index=False)
        print(f"Cluster {cluster_id} telah disimpan ke cluster_{cluster_id}.csv")

    # Function to provide perfume recommendations based on cosine similarity and additional filters
    def recommend_based_on_input(input_scent, input_base_note, input_middle_note, input_ml, input_concentration, input_price):
        # Combine user input aroma into a format similar to the training data
        combined_input = input_scent + ' ' + input_base_note + ' ' + input_middle_note
        input_vector = vectorizer.transform([combined_input])

        # Prediksi cluster untuk input pengguna
        input_cluster = kmeans.predict(input_vector)[0] # [0] karena input pengguna hanya satu (satu vektor), kita mengambil elemen pertama dari array hasil prediksi.
        print(f'Input Cluster: {input_cluster}')

        # Ambil parfum dalam cluster yang sama dengan input pengguna
        cluster_parfumes = data[data['cluster'] == input_cluster]
        cluster_parfumes.to_csv('cluster_parfumes.csv', index=False)

        # Hitung cosine similarity hanya dalam cluster tersebut
        similarity_scores = cosine_similarity(input_vector, vectorizer.transform(cluster_parfumes['combined_notes'])) #cluster_parfumes digunakan untuk mempersempit pencarian/perhitungan kemiripan oleh cosine similarity

        # Urutkan parfum berdasarkan similarity
        similar_parfum_indices = similarity_scores.argsort()[0][::-1]
        filtered_parfumes = cluster_parfumes.iloc[similar_parfum_indices]

        # Filter berdasarkan kriteria tambahan
        filtered_parfumes = filtered_parfumes[(filtered_parfumes['ml'] >= input_ml - 10) & # misal ml = 70, maka yg diambil >= 60 dan <= 80
                                            (filtered_parfumes['ml'] <= input_ml + 10) &
                                            (filtered_parfumes['concentration'].str.contains(input_concentration, case=False)) &
                                            (filtered_parfumes['new_price'] <= input_price)]

        # Ambil 5
        #  rekomendasi teratas
        recommended_parfumes = filtered_parfumes.head(5)

        print(f"\nRekomendasi berdasarkan input Anda:")
        print(recommended_parfumes[['brand', 'name', 'scents', 'base_note', 'middle_note', 'ml', 'concentration', 'new_price']])

        # Export recommendations to CSV
        recommended_parfumes[['brand', 'name', 'scents', 'base_note', 'middle_note', 'ml', 'concentration', 'new_price']].to_csv('recommendation_output.csv', index=False)

        return recommended_parfumes

    # Function to evaluate the recommendation system
    def evaluate_recommendations(recommended_parfumes, relevant_parfumes):
        # Precision@K: Proportion of recommended items that are relevant
        precision_at_k = len(set(recommended_parfumes.index) & set(relevant_parfumes.index)) / len(recommended_parfumes)
        
        # Recall@K: Proportion of relevant items that are recommended
        recall_at_k = len(set(recommended_parfumes.index) & set(relevant_parfumes.index)) / len(relevant_parfumes)
        
        return precision_at_k, recall_at_k

    # Input aroma and additional preferences from the user
    preferred_scent = input("Masukkan aroma yang Anda sukai (misalnya: Woody, Citrus, Floral): ").lower()
    base_note = input("Masukkan base note (misalnya: Oakmoss, Patchouli): ").lower()
    middle_note = input("Masukkan middle note (misalnya: Jasmine, Honey): ").lower()
    ml = float(input("Masukkan kapasitas (ml) yang Anda inginkan: "))
    concentration = input("Masukkan konsentrasi parfum (EDP atau EDT): ").lower()
    price = float(input("Masukkan harga maksimum yang Anda inginkan (dalam dollar): "))

    # Provide recommendations
    recommended_parfumes = recommend_based_on_input(preferred_scent, base_note, middle_note, ml, concentration, price)

def recommendation_form(request):
    if request.method == 'POST':
            form = PerfumePreferenceForm(request.POST)
            if form.is_valid():
                user = form.save()
                auth_login(request, user)
                return redirect('home')
            else:
                print(form.errors)
    else:
        form = PerfumePreferenceForm()
        
        return render(request, 'recommendation_form.html', {'form': form})