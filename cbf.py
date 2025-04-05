import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class CBF:
    @staticmethod
    def load_data(path):
        data = pd.read_csv(path)
        data = data[data["type"].isin(["TV", "Movie", "OVA"])]
        data = data[
            [
                "anime_id",
                "name",
                "genres",
                "licensors",
                "studios",
                "producers",
                "source",
                "rating",
            ]
        ]

        return data

    @staticmethod
    def split_to_list(column):
        return column.dropna().apply(lambda x: x.split(", "))

    @staticmethod
    def encode_data(data):
        mlb = MultiLabelBinarizer()

        genres_encoded = pd.DataFrame(
            mlb.fit_transform(CBF.split_to_list(data["genres"])),
            columns=mlb.classes_,
            index=data.index,
        )

        producers_encoded = pd.DataFrame(
            mlb.fit_transform(CBF.split_to_list(data["producers"])),
            columns=mlb.classes_,
            index=data.index,
        )

        rating_encoded = pd.DataFrame(
            mlb.fit_transform(CBF.split_to_list(data["rating"])),
            columns=mlb.classes_,
            index=data.index,
        )

        studios_encoded = pd.DataFrame(
            mlb.fit_transform(CBF.split_to_list(data["studios"])),
            columns=mlb.classes_,
            index=data.index,
        )

        sources_encoded = pd.DataFrame(
            mlb.fit_transform(CBF.split_to_list(data["source"])),
            columns=mlb.classes_,
            index=data.index,
        )

        animes_encoded = pd.concat(
            [
                data,
                genres_encoded,
                producers_encoded,
                rating_encoded,
                studios_encoded,
                sources_encoded,
            ],
            axis=1,
        )
        animes_encoded.drop(
            columns=[
                "name",
                "anime_id",
                "genres",
                "licensors",
                "producers",
                "rating",
                "studios",
                "source",
            ],
            inplace=True,
        )

        return animes_encoded

    @staticmethod
    def convert_to_matrix(data):
        return csr_matrix(data.values)

    @staticmethod
    def do_cosine_similarity(matrix):
        cosine_sim = cosine_similarity(matrix, dense_output=False)
        return cosine_sim

    @staticmethod
    def convert_to_df(data, path):
        cosine_sim_dense = data.toarray().astype(np.float32)

        animes = CBF.load_data(path)

        cosine_sim_df = pd.DataFrame(
            cosine_sim_dense, index=animes["anime_id"], columns=animes["anime_id"]
        )
        
        if not os.path.exists("model"):
            os.makedirs("model")
            
        if os.listdir("model"):
            for file in os.listdir("model"):
                file_path = os.path.join("model", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        pickle.dump(cosine_sim_df, open("model/anime.pkl", "wb"))

        return cosine_sim_df

    @staticmethod
    def load_model(path):
        return pickle.load(open(path, "rb"))


    @staticmethod
    def recommend_anime(anime_id, cosine_sim, df, top_n=50, page=1, page_size=10):
        if anime_id not in cosine_sim.index:
            return ["Anime ID tidak ditemukan dalam cosine similarity."]

        sim_scores = cosine_sim.loc[anime_id].sort_values(ascending=False)

        sim_scores = sim_scores.iloc[1 : top_n + 1]

        start = (page - 1) * page_size
        end = start + page_size

        recommended_anime = sim_scores.index.tolist()[start:end]

        return recommended_anime

    @staticmethod
    def calculate_precision_by_category(recommended_anime_ids, user_profile_anime_ids, animes, category):
        user_fav_categories = set()
        for anime_id in user_profile_anime_ids:
            values = animes.loc[animes["anime_id"] == anime_id, category].dropna().values
            anime_name = animes.loc[animes["anime_id"] == anime_id, "name"].values
            if len(values) > 0:
                for val in values:
                    user_fav_categories.update([x.strip() for x in val.split(", ")])


        recommend_categories = set()
        for anime_id in recommended_anime_ids:
            values = animes.loc[animes["anime_id"] == anime_id, category].dropna().values
            anime_name = animes.loc[animes["anime_id"] == anime_id, "name"].values
            if len(values) > 0:
                for val in values:
                    recommend_categories.update([x.strip() for x in val.split(", ")])
        print(f"{len(recommend_categories)}")

        print(recommend_categories)

        matched_cat = user_fav_categories.intersection(recommend_categories)

        precision = len(matched_cat) / len(recommend_categories) if recommend_categories else 0

        return precision
    
    @staticmethod
    def calculate_precision_by_rating(recommended_anime_ids, user_profile_anime_ids, animes):
        user_fav_categories = set()
        for anime_id in user_profile_anime_ids:
            values = animes.loc[animes["anime_id"] == anime_id, "rating"].dropna().values
            anime_name = animes.loc[animes["anime_id"] == anime_id, "name"].values
            if len(values) > 0:
                for val in values:
                    user_fav_categories.update([x.strip() for x in val.split(", ")])

        recommend_categories = set()
        for anime_id in recommended_anime_ids:
            values = animes.loc[animes["anime_id"] == anime_id, "rating"].dropna().values
            anime_name = animes.loc[animes["anime_id"] == anime_id, "name"].values
            if len(values) > 0:
                for val in values:
                    recommend_categories.update([x.strip() for x in val.split(", ")])
        print(f"{len(recommend_categories)}")

        matched_cat = user_fav_categories.intersection(recommend_categories)
        precision = len(matched_cat) / len(recommend_categories) if recommend_categories else 0

        return precision
    
    @staticmethod
    def create_user_profile(favorite_anime_ids, history_anime_ids, anime_features_np, animes, fav_weight=0.7, hist_weight=0.3):
        favorite_vectors = []
        history_vectors = []

        animes = animes.reset_index(drop=True)

        for anime_id in favorite_anime_ids:
            if anime_id in animes["anime_id"].values:
                idx = animes.index[animes["anime_id"] == anime_id].tolist()[0]
                if idx < len(anime_features_np):  # Pastikan indeks valid
                    favorite_vectors.append(anime_features_np[idx])
                else:
                    print(f"Index {idx} out of bounds for anime_id {anime_id}")

        for anime_id in history_anime_ids:
            if anime_id in animes["anime_id"].values:
                idx = animes.index[animes["anime_id"] == anime_id].tolist()[0]
                if idx < len(anime_features_np):
                    history_vectors.append(anime_features_np[idx])
                else:
                    print(f"Index {idx} out of bounds for anime_id {anime_id}")

        if favorite_vectors:
            favorite_profile = np.mean(favorite_vectors, axis=0) * fav_weight
        else:
            favorite_profile = np.zeros(anime_features_np.shape[1])

        if history_vectors:
            history_profile = np.mean(history_vectors, axis=0) * hist_weight
        else:
            history_profile = np.zeros(anime_features_np.shape[1])

        user_profile = favorite_profile + history_profile

        return user_profile.reshape(1, -1)
    
    @staticmethod
    def recommend_anime_user_profile_cosine(favorite_anime_ids, history_anime_ids, top_n=50, page=1, page_size=10):
        cosine_sim_df = pd.read_pickle("model/anime.pkl")

        valid_fav_ids = [aid for aid in favorite_anime_ids if aid in cosine_sim_df.index]
        if not valid_fav_ids:
            return []

        similarity_scores = cosine_sim_df.loc[valid_fav_ids].mean(axis=0)

        sorted_similarities = similarity_scores.sort_values(ascending=False)

        user_watched_anime_ids = set(favorite_anime_ids) | set(history_anime_ids)
        
        recommendations = [
            {"anime_id": int(anime_id), "similarity_score": float(score)}
            for anime_id, score in sorted_similarities.items()
            if anime_id not in user_watched_anime_ids
        ][:top_n]  # Ambil top_n rekomendasi

        start = (page - 1) * page_size
        end = start + page_size
        paginated_recommendations = recommendations[start:end]

        return paginated_recommendations
