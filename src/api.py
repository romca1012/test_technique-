from fastapi import FastAPI, HTTPException, Query
from .recommender import ReviewRecommender

app = FastAPI(title="SensCritique - Similar Reviews (same movie)")

reco = ReviewRecommender()

@app.get("/similar")
def similar_reviews(
    review_id: str = Query(..., description="ID de la critique"),
    k: int = Query(5, ge=1, le=20)
):
    try:
        results = reco.similar(review_id=review_id, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "review_id": review_id,
        "top_k": k,
        "results": results
    }
