    import pandas as pd
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity

    df = pd.read_csv("dataset_final.csv")
    
    tfidf = joblib.load("tfidf.pkl")

    item_profile = joblib.load("item_profile.pkl")

    def get_rekomendasi(menu, flavor, price, dine, rating, top_n=10):

    # =====================================================
    # 1. USER QUERY
    # =====================================================
    
    user_query = f"{menu} {flavor}"
    user_vec = tfidf.transform([user_query])


    df = df_menu.copy()
    # =====================================================
    # 2. TF-IDF SIMILARITY
    # =====================================================

    df["tfidf_similarity"] = cosine_similarity(
        item_profile,
        user_vec
    ).flatten()

    # =====================================================
    # 3. FEATURE SCORING (0-1 SCALE)
    # =====================================================

    df["rating_score"] = df["avgRating"] / 5.0

    df["price_score"] = (
        df["range_price"] == price
    ).astype(int)

    if dine == "both":
        df["dine_score"] = 1
    else:
        df["dine_score"] = df["dine_option"].apply(
            lambda x: 1 if x == dine or x == "both" else 0
        )

    # =====================================================
    # 4. COMBINED SIMILARITY SCORE
    # =====================================================

    df["similarity"] = (
        df["tfidf_similarity"] +
        df["rating_score"] +
        df["price_score"] +
        df["dine_score"]
    ) / 4

    # =====================================================
    # 5. FILTER OUT NON-RELEVANT ITEMS
    # =====================================================

    df = df[df["tfidf_similarity"] > 0]

    # =====================================================
    # 6. FILTER FUNCTION
    # =====================================================

    def apply_filter(df, use_price=True, use_dine=True, use_rating=True):

        d = df.copy()

        if use_price:
            d = d[d["range_price"] == price]

        if use_dine and dine != "both":
            d = d[
                (d["dine_option"] == dine) |
                (d["dine_option"] == "both")
            ]

        if use_rating:
            d = d[d["avgRating"] >= rating]

        return d

    # =====================================================
    # 7. FALLBACK LEVELS
    # =====================================================

    levels = [

        ({"use_price": True,  "use_dine": True,  "use_rating": True},
         "Exact Match"),

        ({"use_price": False, "use_dine": True,  "use_rating": True},
         "Rekomendasi Alternatif Harga"),

        ({"use_price": False, "use_dine": False, "use_rating": True},
         "Rekomendasi Alternatif"),

        ({"use_price": False, "use_dine": False, "use_rating": False},
         "Menu Serupa"),
    ]

    # =====================================================
    # 8. BUILD RESULT (FALLBACK SYSTEM)
    # =====================================================

    result = pd.DataFrame()

    for kwargs, label in levels:

        temp = apply_filter(df, **kwargs)

        if len(temp) == 0:
            continue

        temp = temp.sort_values(
            by=["similarity", "tfidf_similarity", "avgRating"],
            ascending=[False, False, False]
        )

        temp = temp.drop_duplicates(subset="nama_tempat")

        temp["recommendation_type"] = label

        result = pd.concat([result, temp], ignore_index=True)

        result = result.drop_duplicates(subset="nama_tempat")

        if len(result) >= top_n:
            break

    # =====================================================
    # 9. HANDLE EMPTY RESULT
    # =====================================================

    if result.empty:
        return pd.DataFrame({
            "message": ["Menu serupa tidak ditemukan"]
        })

    # =====================================================
    # 10. FINAL OUTPUT
    # =====================================================

    result = result.head(top_n).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result[[
        "rank",
        "nama_tempat",
        "recommended_menu",
        "avgRating",
        "range_price",
        "dine_option",
        "tfidf_similarity",
        "similarity",
        "recommendation_type"
    ]]
