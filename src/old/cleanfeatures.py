import pandas as pd

foundersData = pd.read_csv("/home/andri/conspiracy-founders/data/interim/foundersData_FINAL.csv", index_col=0)
foundersFeatures = pd.read_csv("/home/andri/conspiracy-founders/data/interim/foundersFeatures_FINAL.csv", index_col=0)

foundersData = foundersData.reset_index(drop=True)
foundersFeatures = foundersFeatures.reset_index(drop=True)

foundersData.to_csv("/home/andri/conspiracy-founders/data/interim/foundersData_FINAL2.csv")
foundersFeatures.to_csv("/home/andri/conspiracy-founders/data/interim/foundersFeatures_FINAL2.csv")