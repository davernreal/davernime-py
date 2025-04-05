# Anime Recommendation API

This API provides endpoints to upload anime datasets, retrieve anime-based recommendations, and get personalized suggestions based on user data.

---

## 📍 Endpoints

### `GET /`
Returns a simple welcome message to indicate that the API is running.

#### Response
```json
{
  "message": "Welcome to the Anime Recommendation API!"
}
```

---

### `POST /anime`
Uploads an anime dataset and updates the cosine similarity matrix used for recommendations.

#### Request Body (multipart/form-data)
| Field    | Type | Description                       |
|----------|------|-----------------------------------|
| dataset  | file | A CSV file containing anime data. |

#### Responses
- `200 OK`: Dataset uploaded and model processed successfully.
- `400 Bad Request`: File missing, invalid, or exceeds size limit.

---

### `GET /anime/<int:anime_id>`
Fetches anime recommendations for a specific anime ID using cosine similarity.

#### Query Parameters
| Parameter  | Type | Default | Description                         |
|------------|------|---------|-------------------------------------|
| page       | int  | 1       | The current page number.            |
| page_size  | int  | 10      | Number of results per page.         |

#### Responses
- `200 OK`: Recommendations retrieved successfully.
- `404 Not Found`: Anime ID not found in dataset.
- `500 Internal Server Error`: Error occurred during recommendation process.

#### Example Response
```json
{
  "page": 1,
  "page_size": 10,
  "recommendations": [21881, 50275, 30347, ...],
  "precision": 0.45,
  "success": true
}
```

---

### `POST /anime/user`
Generates personalized anime recommendations based on user data.

#### Request Body (JSON)
```json
{
  "user_favorites": [1, 2, 3],
  "user_history": [4, 5, 6]
}
```

#### Responses
- `200 OK`: Recommendations based on the user's profile returned successfully.

---

## 📦 Example CSV Format for Upload
Your uploaded CSV file should include structured anime data, for example:

| anime_id | title        | genres           |
|----------|--------------|------------------|
| 1        | Naruto       | Action, Adventure |
| 2        | One Piece    | Action, Fantasy   |

or

You can download the dataset [here](https://drive.google.com/file/d/1T9wqnBjRMpWha4e3VsOKvKIFWnloXV8O/view?usp=sharing "here")
Then put that thing manually into dataset folder `dataset/anime.csv`

---

## 🚀 How to Run

1. Clone the repository  
2. Make sure Python and dependencies are installed  
3. Run the Flask app:
```bash
flask --app main run
```
If yout want to debugging, run
```bash
flask --app main run --debug
```
