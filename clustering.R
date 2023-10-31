df <- read.csv("C:/Users/yayag/Kaggle_SCP/data/temp/B_cells_normalized_filledna.csv")
dataframe <- as.data.frame(df)
row.names(dataframe) <- dataframe[,1]
dataframe <- dataframe[,-1]

d <- dist(df)
hc <- hclust(d)
plot(hc)
