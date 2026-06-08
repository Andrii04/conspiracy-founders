import pandas as pd


def splitDataset():
    df_features = pd.read_csv("data/interim/foundersFeatures_FINAL.csv", index_col=0)

    df_anchors = df_features.sample(frac=0.25, random_state=42)
    df_candidates = df_features.drop(df_anchors.index)

    # meglio invece di sample, prendere i primi più alti nella scala conspiracy (quindi non randomico)
    # sort su colonna conspiracy e poi .head della dimensione che vogliamo
    df_anchors = df_anchors.reset_index(drop=True)
    df_candidates = df_candidates.reset_index(drop=True)

    print("--- STATISTICHE SPLIT DI TEST ---")
    print(f"Numero di Anchors (Casi target):   {df_anchors.shape[0]}")
    print(f"Numero di Candidates (Pool controllo): {df_candidates.shape[0]}")
    print(f"Totale utenti verificati:          {df_anchors.shape[0] + df_candidates.shape[0]}")

    df_anchors.to_csv("/home/andri/conspiracy-founders/data/interim/anchors_test.csv")
    df_candidates.to_csv("/home/andri/conspiracy-founders/data/interim/candidates_test.csv")

if __name__ == "__main__":
    splitDataset()