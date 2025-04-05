from flask import Flask, jsonify, request, make_response
from cbf import CBF
import os
import utils
import numpy as np

app = Flask(__name__)
DATASET_PATH = "dataset/anime.csv"
DATASET_ALLOWED_TYPE = {"csv"}
UPLOAD_DATASET = "dataset/"

app.config["UPLOAD_DATASET"] = UPLOAD_DATASET
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

"""
- GET /
    Returns a welcome message for the API.
"""

@app.route("/")
def hello():
    data = {"message": "Welcome to AnimeVerse API"}
    return jsonify(data), 200


"""
- GET /anime
    Returns a placeholder response indicating the endpoint is under construction.
"""

@app.route("/anime")
def anime():
    return make_response("Lu Wibu", 400)


"""
- POST /anime
    Uploads a dataset file, validates it, updates the cosine similarity matrix, 
    and saves the file to the server.
    Request Data:
        - dataset: A file (CSV) containing anime data.
    Responses:
        - 200: If the dataset is successfully uploaded and processed.
        - 400: If the file is missing, invalid, or exceeds the size limit.
"""


@app.route("/anime", methods=["POST"])
def post_anime():
    def update_cosine_similarity():
        try:
            data = CBF.load_data(DATASET_PATH)
            print("done loading data")

            encoded_data = CBF.encode_data(data)
            print("done encoding data")

            matrix = CBF.convert_to_matrix(encoded_data)
            print("done converting to matrix")

            similarity_matrix = CBF.do_cosine_similarity(matrix)
            print("done similarity matrix")

            cosine_df = CBF.convert_to_df(similarity_matrix, DATASET_PATH)
            print("done converting to df")

            return True

        except Exception as e:
            raise e

    try:
        if "dataset" not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400

        file = request.files["dataset"]

        if not file or not utils.allowed_file(file.filename, DATASET_ALLOWED_TYPE):
            return jsonify({"success": False, "message": "Invalid file type"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > app.config["MAX_CONTENT_LENGTH"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "File size exceeds the maximum limit (16 MB)",
                    }
                ),
                400,
            )
        file.seek(0)

        extension = file.filename.rsplit(".", 1)[1].lower()
        filename = f"anime.{extension}"
        file.save(os.path.join(app.config["UPLOAD_DATASET"], filename))

        success = update_cosine_similarity()
        return jsonify(
            {
                "success": success,
                "message": (
                    "Data updated successfully"
                    if success
                    else "Failed to update cosine similarity"
                ),
            }
        )

    except Exception as e:
        return make_response(jsonify({"status": "error", "message": e}))


"""
- GET /anime/<int:anime_id>
    Retrieves anime recommendations based on a specific anime ID.
    Query Parameters:
        - page (int, optional): The page number for pagination. Default is 1.
        - page_size (int, optional): The number of recommendations per page. Default is 10.
    Responses:
        - 200: If recommendations are successfully retrieved.
        - 404: If the anime ID is not found in the dataset.
        - 500: If an error occurs during processing.
"""


@app.route("/anime/<int:anime_id>")
def get_anime(anime_id):
    model = CBF.load_model("model/anime.pkl")
    animes = CBF.load_data(DATASET_PATH)

    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=10, type=int)

    try:
        if anime_id not in animes["anime_id"].values:
            return (
                jsonify({"success": False, "message": "Anime ID tidak ditemukan."}),
                404,
            )
        recommendations = CBF.recommend_anime(
            anime_id, model, animes, page=page, page_size=page_size
        )
        precision_score_genre = CBF.calculate_precision_by_category(
            recommendations, [anime_id], animes, "genres"
        )
        precision_score_studio = CBF.calculate_precision_by_category(
            recommendations, [anime_id], animes, "studios"
        )
        precision_score_producer = CBF.calculate_precision_by_category(
            recommendations, [anime_id], animes, "producers"
        )
        precision_score_source = CBF.calculate_precision_by_category(
            recommendations, [anime_id], animes, "source"
        )
        precision_score_rating = CBF.calculate_precision_by_rating(
            recommendations, [anime_id], animes
        )

        total_precision = (
            precision_score_genre
            + precision_score_studio
            + precision_score_producer
            + precision_score_rating
            + precision_score_source
        ) / 5

        return jsonify(
            {
                "success": True,
                "recommendations": recommendations,
                "page": page,
                "page_size": page_size,
                "precision": total_precision,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


"""
- POST /anime/user
    Retrieves anime recommendations based on a user profile.
    Request Data:
        - user_favorites: An array of anime IDs representing the user's favorite anime.
        - user_history: An array of anime IDs representing the user's viewing history.
    Responses:
        - 200: If recommendations are successfully retrieved.
"""
@app.route("/anime/user", methods=["POST"])
def post_anime_user():
    model = CBF.load_model("model/anime.pkl")
    animes = CBF.load_data(DATASET_PATH)

    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=10, type=int)
    try:
        import json

        animes_encoded = CBF.encode_data(animes)
        user_favorites = json.loads(request.form.get("user_favorites", "[]"))
        user_history = json.loads(request.form.get("user_history", "[]"))
        
        if "user_favorites" not in request.form or "user_history" not in request.form:
            return make_response(jsonify({"message": "Invalid request data"}))

        user_profile = CBF.create_user_profile(
            user_favorites, user_history, np.array(animes_encoded, dtype='float32'), animes
        )
        
        recommendations = CBF.recommend_anime_user_profile_cosine(
            user_favorites, user_history, page=page, page_size=page_size
        )
        
        try:
            user_recommendation = [
                {
                    "id": int(i["anime_id"]), 
                    "similarity_score": float(i["similarity_score"]) 
                }
                for i in recommendations
            ]
        except (TypeError, KeyError) as e:
            return jsonify({"success": False, "message": f"Error processing recommendations: {str(e)}"}), 500

        return make_response(
            jsonify(
                {
                    "success": True,
                    "recommendations": user_recommendation,
                    "page": page,
                    "page_size": page_size,
                }
            )
        )

    except Exception as e:
        print("ERROR")
        print(e)
        return jsonify({"success": False, "error": e})
