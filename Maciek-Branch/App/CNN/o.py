import torch

# Przykładowy tensor z wartościami NaN
tensor_with_nan = torch.tensor([[1.0, 2.0, 3],
                                [4.0, float('nan'), 6.0],
                                [float('nan'), 8.0, 9.0]])

# Znajdź indeksy wartości NaN
nan_indices = torch.isnan(tensor_with_nan).any(dim=2)

# Wybierz tylko te wiersze, które nie zawierają wartości NaN
tensor_without_nan = tensor_with_nan[~nan_indices]

print("Tensor bez NaN:")
print(tensor_without_nan)