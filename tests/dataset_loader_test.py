from ionyx.datasets import DataSetLoader

print('Beginning dataset loader test...')

data, X, y1, y2 = DataSetLoader.load_bike_sharing()
print("Bike-sharing data:")
print("Data shape = {0}, X shape = {1}, y1 shape = {2}, y2 shape = {3}"
      .format(data.shape, X.shape, y1.shape, y2.shape))
print(data.head())
print("")

data, X, y = DataSetLoader.load_forest_cover()
print("Forest cover data:")
print("Data shape = {0}, X shape = {1}, y shape = {2}"
      .format(data.shape, X.shape, y.shape))
print(data.head())
print("")

data, X, y = DataSetLoader.load_otto_group()
print("Otto group data:")
print("Data shape = {0}, X shape = {1}, y shape = {2}"
      .format(data.shape, X.shape, y.shape))
print(data.head())
print("")

data, X, y = DataSetLoader.load_property_inspection()
print("Property inspection data:")
print("Data shape = {0}, X shape = {1}, y shape = {2}"
      .format(data.shape, X.shape, y.shape))
print(data.head())
print("")

data, X, y, = DataSetLoader.load_time_series()
print("Time series data:")
print("Data shape = {0}, X shape = {1}, y shape = {2}"
      .format(data.shape, X.shape, y.shape))
print(data.head())
print("")

print('Done.')
