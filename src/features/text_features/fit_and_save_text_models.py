from loaders.load_notes import load_notes
from text_model_pipeline import text_model_pipeline

df = load_notes()

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        df=df,
        max_features=500,
        max_df=0.90,
        min_df=10,
        ngram_range=(1, 3),
    )
    print("Done!")
