import os

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH,"data")
CLEANSETS_PATH = os.path.join(BASE_PATH, "cleaned_sets")
SUBMISSION_PATH = os.path.join(BASE_PATH, "submissions")

FILENAMES = {
    #INPUT
    "TRAIN_SALES" : os.path.join(DATA_PATH, "sales_train.csv"),
    "TEST_SALES" : os.path.join(DATA_PATH, "test.csv"),
    "SAMPLE_SUBM" : os.path.join(DATA_PATH, "sample_submission.csv"),
    "SHOPS" : os.path.join(DATA_PATH, "shops.csv"),
    "ITEMS" : os.path.join(DATA_PATH, "items.csv"),
    "ITEM_CATEGORIES" : os.path.join(DATA_PATH, "item_categories.csv"),
    #OUTPUT
    "X_TRAIN": os.path.join(CLEANSETS_PATH, "X_train.csv")
}