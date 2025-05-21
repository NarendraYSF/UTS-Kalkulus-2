from surprise import SVD, Dataset, Reader
import pandas as pd

# Create a small dataset
ratings_dict = {
    'user_id': [1, 1, 2, 2],
    'item_id': [1, 2, 1, 2],
    'rating': [5, 3, 2, 4]
}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.25)

# Instantiate and train the model
model = SVD(n_factors=5, n_epochs=5, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Test the model
predictions = model.test(testset)
print(predictions)
