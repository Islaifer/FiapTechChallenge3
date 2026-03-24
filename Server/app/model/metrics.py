class Metrics:
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    rmse = 0
    mae = 0
    silhouette = 0

    def __init__(self, accuracy, precision, recall, f1, rmse, mae, silhouette):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.rmse = rmse
        self.mae = mae
        self.silhouette = silhouette