from loaders.load_notes import load_notes
from text_model_pipeline import text_model_pipeline

df = load_notes(nrows=300)

if __name__ == "__main__":
    text_model_pipeline(
        model="tfidf",
        df=df,
        max_features=200,
        max_df=0.95,
        min_df=10,
        ngram_range=(1, 3),
    )
