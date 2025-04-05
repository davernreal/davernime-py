def allowed_file(filename, DATASET_ALLOWED_TYPE):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in DATASET_ALLOWED_TYPE