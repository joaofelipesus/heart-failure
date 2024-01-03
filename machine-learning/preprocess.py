import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

class PreprocessPipeline:
    ORDINAL_CATEGORIES = ['Sex', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    ONE_HOT_ENCODED_CATEGORIES = ['ChestPainType']
    COLUMNS_TO_DROP = ['ChestPainType']
    CACHE_PATH = 'machine-learning/cache'

    def __init__(self, dataset_path):
        self._df = pd.read_csv(dataset_path)
        self._ordinal_encoder = OrdinalEncoder()
        self._one_hot_encoder = OneHotEncoder()
        self._scaler = MinMaxScaler()

    def fit_transform(self):
        self._ordinal_encoder.fit(self._df[self.ORDINAL_CATEGORIES])
        self._one_hot_encoder.fit(self._df[self.ONE_HOT_ENCODED_CATEGORIES])
        self._encode_ordinal_features()
        self._encode_one_hot_features()
        self._drop_columns()
        X = self._normalize_features()
        y = self._df['HeartDisease'].values
        self._save_encoders_and_scaler()
        return X, y

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

    def _save_encoders_and_scaler(self):
        pickle.dump(self._ordinal_encoder, open(f'{self.CACHE_PATH}/ordinal_encoder.pkl', 'wb'))
        pickle.dump(self._one_hot_encoder, open(f'{self.CACHE_PATH}/one_hot_encoder.pkl', 'wb'))
        pickle.dump(self._scaler, open(f'{self.CACHE_PATH}/scaler.pkl', 'wb'))

    def _normalize_features(self):
        X = self._df.drop(columns=['HeartDisease']).values
        self._scaler.fit(X)
        return self._scaler.transform(X)


if __name__ == '__main__':
    DATASET_PATH = 'data/train-heart.csv'
    preprocess_pipeline = PreprocessPipeline(DATASET_PATH)
    X, y = preprocess_pipeline.fit_transform()
    print(X.shape)
    print(y.shape)
