import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class PreprocessPipeline:
    ORDINAL_CATEGORIES = ['Sex', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    ONE_HOT_ENCODED_CATEGORIES = ['ChestPainType']
    COLUMNS_TO_DROP = ['ChestPainType']

    def __init__(self, dataset_path):
        self._df = pd.read_csv(dataset_path)
        self._ordinal_encoder = OrdinalEncoder()
        self._one_hot_encoder = OneHotEncoder()

    def fit_transform(self):
        self._ordinal_encoder.fit(self._df[self.ORDINAL_CATEGORIES])
        self._one_hot_encoder.fit(self._df[self.ONE_HOT_ENCODED_CATEGORIES])
        self._encode_ordinal_features()
        self._encode_one_hot_features()
        self._drop_columns()
        return self._df

    def _encode_ordinal_features(self):
        self._df[self.ORDINAL_CATEGORIES] = self._ordinal_encoder.transform(
            self._df[self.ORDINAL_CATEGORIES]
        )

    def _encode_one_hot_features(self):
        one_hot_encoded_features = self._one_hot_encoder.transform(
            self._df[self.ONE_HOT_ENCODED_CATEGORIES]
        )
        one_hot_encoded_df = pd.DataFrame(
            data=one_hot_encoded_features.toarray(),
            columns=self._one_hot_encoder.get_feature_names_out()
        )
        self._df = pd.concat([self._df, one_hot_encoded_df], axis=1)

    def _drop_columns(self):
        self._df = self._df.drop(columns=self.COLUMNS_TO_DROP)

    # TODO: use pickle to save encoders after fit, it will be used on the API and predict pipeline.
    def _save_encoders(self):
        pass


if __name__ == '__main__':
    DATASET_PATH = 'data/train-heart.csv'
    preprocess_pipeline = PreprocessPipeline(DATASET_PATH)
    df = preprocess_pipeline.fit_transform()
    print(df.head())
