import numpy as np
from typing import Optional, Literal, Union, List
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class GridWorldEncoder:
    def __init__(self, categories: Union[np.ndarray, list], 
                 mode: Literal['pomdp', 'directions', 'mdp'], 
                 collision_hint: Optional[bool] = True,
                 encoder: Literal['label', 'one-hot'] = 'one-hot'):
        self.mode = mode
        self.collision_hint = collision_hint
        self.encoder_type = encoder
        if self.mode == 'mdp':
            self.len_room = int(len(categories) ** 0.5)
        # Фильтрация категорий при необходимости
        self.categories = np.array(categories).reshape(-1, 1)
        if not collision_hint and self.mode == 'pomdp':
            self.categories = self.categories[self.categories >= 0]

        # Подготовка энкодеров в зависимости от режима и типа кодирования
        if encoder == 'one-hot':
            self.encoder = OneHotEncoder(categories=[self.categories.ravel()], sparse_output=False)
            self.encoder.fit(self.categories)
        else:  # label
            self.encoder = LabelEncoder()
            self.encoder.fit(self.categories.ravel())
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        # Приведение данных к правильной форме
        if self.mode == 'mdp' and data.shape[-1] == 2:
            data = self.len_room * data[..., 0] + data[..., 1]

        data = data.reshape(-1, 1)
        # Определение размера выходных данных
        if self.encoder_type == 'one-hot':
            encoded_size = len(self.categories)
        else:
            encoded_size = 1
        
        # Создание результата с NaN
        result = np.full((data.shape[0], encoded_size), np.nan, dtype=np.float32)
        
        # Поиск не-NaN строк
        non_nan_mask = ~np.isnan(data).any(axis=1)
        if non_nan_mask.ndim > 1:
            non_nan_mask = non_nan_mask.squeeze()
        non_nan_data = data[non_nan_mask]
        # Обработка пустых данных
        if non_nan_data.size == 0:
            return result
        
        # Преобразование не-NaN данных
        if self.encoder_type == 'one-hot':
            encoded = self.encoder.transform(non_nan_data.reshape(-1, 1))
            result[non_nan_mask] = encoded
        else:  # label
            encoded = self.encoder.transform(non_nan_data.ravel()).reshape(-1, 1)
            result[non_nan_mask] = encoded
        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # Определение формы результата
        result_shape = (data.shape[0], 1)
        result = np.full(result_shape, np.nan)
        
        # Поиск не-NaN строк
        if self.encoder_type == 'one-hot':
            non_nan_mask = ~np.isnan(data).any(axis=1)
        else:  # label
            non_nan_mask = ~np.isnan(data).all(axis=1)
        
        non_nan_data = data[non_nan_mask]
        
        # Обработка пустых данных
        if non_nan_data.size == 0:
            return result
        
        # Обратное преобразование
        if self.encoder_type == 'one-hot':
            decoded = self.encoder.inverse_transform(non_nan_data)
            result[non_nan_mask] = decoded
        else:  # label
            decoded = self.encoder.inverse_transform(non_nan_data.ravel())
            result[non_nan_mask, 0] = decoded
        
        if self.mode == 'mdp':
            result = np.stack((result // self.len_room, result % self.len_room), axis=-1).squeeze()

        return result
    
class SimpleOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, max_categories: int):
        
        self.max_categories = max_categories
        self.categories = {}
    
    def fit(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]]) -> None:
        
        if isinstance(x, list):
            x = np.array(x)
        
        unique_values = np.unique(x)

        for value in unique_values:
            self.categories[int(value)] = len(self.categories)
        
        return self

    def transform(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]]) -> NDArray:
        
        if isinstance(x, list):
            x = np.array(x)
        
        n_categories = np.unique(x).shape[0]
    
        if n_categories > self.max_categories:
            raise RuntimeError(f'The number of unique observations' \
                            f'{n_categories} is greater than the maximum allowed {self.max_categories}')

        result = np.zeros((x.shape[0], self.max_categories), dtype=np.float32)

        for i, value in enumerate(x):

            if int(value) not in self.categories.keys():
                self.categories[int(value)] = len(self.categories)
            
            result[i, self.categories[value]] = 1.0
        
        return result