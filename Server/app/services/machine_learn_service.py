import shap
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, silhouette_score
from app.model.flight import FlightClassifier, FlightRegression
from app.model.metrics import Metrics

class MachineLearningService:
    def __init__(self):
        self.silhouette = 0
        self.rmse = 0
        self.mae = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

        self._classifier_model = None
        self._regression_model = None
        self._cluster_model = None

        self.airports = None

        self.lock = asyncio.Lock()
        self.RUNNING = False
        self.executor = None

        self.airline_encoder = None
        self.season_encoder = None
        self.scaler = None
        self.route_stats = None

        self.regression_feature_columns = None
        self.classifier_feature_columns = None

    async def get_metrics(self):
        return Metrics(self.accuracy, self.precision, self.recall, self.f1, self.rmse, self.mae, self.silhouette)

    def cluster(self, route_stats):
        route_stats = route_stats[route_stats["COUNT"] > 5].copy()

        route_stats.loc[:, "DELAY_STD"] = route_stats["DELAY_STD"].clip(lower=0)
        route_stats.loc[:, "DELAY_MEAN"] = route_stats["DELAY_MEAN"].clip(lower=0)

        route_stats.loc[:,"DELAY_MEAN_LOG"] = np.log1p(route_stats["DELAY_MEAN"])
        route_stats.loc[:,"DELAY_STD_LOG"] = np.log1p(route_stats["DELAY_STD"])
        route_stats.loc[:,"DISTANCE_LOG"] = np.log1p(route_stats["DISTANCE"])

        route_stats = route_stats.replace([np.inf, -np.inf], 0)
        route_stats = route_stats.fillna(0)

        x = route_stats[["DELAY_MEAN_LOG", "DELAY_STD_LOG", "DISTANCE_LOG"]]
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)

        self._cluster_model = KMeans(n_clusters=2, random_state=7)
        route_stats["ROUTE_CLUSTER"] = self._cluster_model.fit_predict(x_scaled)

        plt.scatter(
            route_stats["DELAY_MEAN"],
            route_stats["DELAY_STD"],
            c=route_stats["ROUTE_CLUSTER"]
        )

        plt.xlabel("Delay médio")
        plt.ylabel("Desvio do delay")
        plt.title("Rotas caóticas")
        plt.savefig("../Data/Plots/rote_cluster.png")

        score = silhouette_score(x, route_stats["ROUTE_CLUSTER"])

        print("============================================")
        print('A métrica do modelo de cluster é: ')
        print(f'Silhouette = {score}')
        print("============================================")

        self.silhouette = score

        self.route_stats = route_stats[["ROUTE", "DELAY_MEAN", "DELAY_STD", "COUNT", "ROUTE_CLUSTER"]]

        return route_stats


    def regression_model(self, data):
        print('Creating regression model....')
        x = data.drop("DEPARTURE_DELAY", axis=1)
        y = data["DEPARTURE_DELAY"]

        y_log = np.log1p(y)

        x_train, x_test, y_train, y_test = train_test_split(x,y_log,test_size=0.2,random_state=7)

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=7
        )

        model.fit(x_train, y_train)
        predict_values = model.predict(x_test)
        predict_values = np.expm1(predict_values)
        y_test_original = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_test_original, predict_values))
        mae = mean_absolute_error(y_test_original, predict_values)

        print("============================================")
        print('As métricas do modelo de regressão são: ')
        print(f'RMSE = {rmse}')
        print(f'MAE = {mae}')
        print("============================================")

        explainer = shap.Explainer(model)
        shap_values = explainer(x_test.sample(1000))

        shap.summary_plot(shap_values, x_test.sample(1000), show=False)
        plt.savefig("../Data/Plots/regression_shap.png")

        self.rmse = rmse
        self.mae = mae

        self.regression_feature_columns = x.columns.tolist()

        return model

    def classifier_model(self, data):
        print('Creating classifier model....')
        x = data.drop("HAD_DELAY", axis=1)
        y = data["HAD_DELAY"]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=7)

        model = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            scale_pos_weight=1.7,
            gamma=0.1,
            random_state=7
        )

        model.fit(x_train, y_train)
        predict_values = model.predict(x_test)

        accuracy = accuracy_score(y_test, predict_values) * 100
        precision = precision_score(y_test, predict_values) * 100
        recall = recall_score(y_test, predict_values) * 100
        f1 = f1_score(y_test, predict_values) * 100

        print("============================================")
        print('As métricas do modelo de classificação são: ')
        print(f'Accuracy = {accuracy}%')
        print(f'Precision = {precision}%')
        print(f'Recall = {recall}%')
        print(f'F1Score = {f1}%')
        print("============================================")

        explainer = shap.Explainer(model)
        shap_values = explainer(x_test.sample(1000))

        shap.summary_plot(shap_values, x_test.sample(1000), show=False)
        plt.savefig("../Data/Plots/classifier_shap.png")

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

        self.classifier_feature_columns = x.columns.tolist()

        return model

    def analysis_task(self):
        print('Carregando dados...')
        #airlines = pd.read_csv('../../Data/airlines.csv')
        self.airports = pd.read_csv('../Data/airports.csv')
        flights = pd.read_csv('../Data/flights.csv', low_memory=False)
        print('Dados carregados')

        print('Preparando dados para análise de atrasos...')
        print('Excluindo voos cancelados...')
        flights_delays = flights[flights['CANCELLED'] != 1].copy()
        drop_cols = ['CANCELLED', 'CANCELLATION_REASON', 'YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'DEPARTURE_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
        flights_delays = flights_delays.drop(drop_cols, axis=1)
        print('Voos cancelados removidos com sucesso')

        print('Preparando tipos de atrasos...')
        flights_delays = flights_delays.fillna(0)

        delay_cols = [
            "WEATHER_DELAY",
            "AIRLINE_DELAY",
            "AIR_SYSTEM_DELAY",
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY"
        ]

        for col in delay_cols:
            flights_delays.loc[:, "HAS_" + col] = (flights_delays[col] > 0).astype(int)

        flights_delays.loc[:, "HAD_DELAY"] = (flights_delays["DEPARTURE_DELAY"] > 0).astype(int)

        print('Tipos de atrasos preparados com sucesso')

        print('Criando variáveis de tempo em minutos e seno e cosseno para melhor análise...')
        flights_delays.loc[:,"DEP_MINUTES"] = flights_delays["SCHEDULED_DEPARTURE"].apply(
            lambda x: (x // 100) * 60 + (x % 100)
        )

        flights_delays.loc[:,"DEP_SIN"] = np.sin(2 * np.pi * flights_delays["DEP_MINUTES"] / 1440)
        flights_delays.loc[:,"DEP_COS"] = np.cos(2 * np.pi * flights_delays["DEP_MINUTES"] / 1440)

        conditions = [
            ((flights_delays["MONTH"] == 3) & (flights_delays["DAY"] >= 20)) |
            (flights_delays["MONTH"].isin([4,5])) |
            ((flights_delays["MONTH"] == 6) & (flights_delays["DAY"] <= 20)),

            ((flights_delays["MONTH"] == 6) & (flights_delays["DAY"] >= 21)) |
            (flights_delays["MONTH"].isin([7,8])) |
            ((flights_delays["MONTH"] == 9) & (flights_delays["DAY"] <= 22)),

            ((flights_delays["MONTH"] == 9) & (flights_delays["DAY"] >= 23)) |
            (flights_delays["MONTH"].isin([10,11])) |
            ((flights_delays["MONTH"] == 12) & (flights_delays["DAY"] <= 20))
        ]

        choices = ["Spring", "Summer", "Fall"]

        flights_delays.loc[:,"SEASON"] = np.select(conditions, choices, default="Winter")

        self.season_encoder = LabelEncoder()
        flights_delays.loc[:, "SEASON_ENCODED"] = self.season_encoder.fit_transform(flights_delays["SEASON"])

        flights_delays.loc[:,"MONTH_SIN"] = np.sin(2 * np.pi * flights_delays["MONTH"] / 12)
        flights_delays.loc[:,"MONTH_COS"] = np.cos(2 * np.pi * flights_delays["MONTH"] / 12)

        days_in_month = flights_delays["MONTH"].map({
            1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
            7:31, 8:31, 9:30, 10:31, 11:30, 12:31
        })

        flights_delays.loc[:,"DAY_SIN"] = np.sin(2 * np.pi * flights_delays["DAY"] / days_in_month)
        flights_delays.loc[:,"DAY_COS"] = np.cos(2 * np.pi * flights_delays["DAY"] / days_in_month)

        flights_delays.loc[:,"DAY_OF_WEEK_SIN"] = np.sin(2 * np.pi * flights_delays["DAY_OF_WEEK"] / 7)
        flights_delays.loc[:,"DAY_OF_WEEK_COS"] = np.cos(2 * np.pi * flights_delays["DAY_OF_WEEK"] / 7)

        print('Variável de tempo em seno e cosseno criadas com sucesso')

        print('Criando variáveis de localização...')

        lookup = self.airports.set_index("IATA_CODE")

        flights_delays.loc[:,"ORIGIN_LAT"] = flights_delays["ORIGIN_AIRPORT"].map(lookup["LATITUDE"])
        flights_delays.loc[:,"ORIGIN_LON"] = flights_delays["ORIGIN_AIRPORT"].map(lookup["LONGITUDE"])

        flights_delays.loc[:,"DESTINATION_LAT"] = flights_delays["DESTINATION_AIRPORT"].map(lookup["LATITUDE"])
        flights_delays.loc[:,"DESTINATION_LON"] = flights_delays["DESTINATION_AIRPORT"].map(lookup["LONGITUDE"])

        flights_delays.loc[:,"DELTA_LAT"] = flights_delays["DESTINATION_LAT"] - flights_delays["ORIGIN_LAT"]
        flights_delays.loc[:,"DELTA_LON"] = flights_delays["DESTINATION_LON"] - flights_delays["ORIGIN_LON"]

        lat1 = np.radians(flights_delays["ORIGIN_LAT"])
        lon1 = np.radians(flights_delays["ORIGIN_LON"])
        lat2 = np.radians(flights_delays["DESTINATION_LAT"])
        lon2 = np.radians(flights_delays["DESTINATION_LON"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        flights_delays.loc[:,"DISTANCE_HAVERSINE"] = 6371 * c

        flights_delays.loc[:, "ROUTE"] = flights_delays["ORIGIN_AIRPORT"] + "_" + flights_delays["DESTINATION_AIRPORT"]

        print('Variáveis de localização criadas')

        print('Criando variáveis das companhias aéreas...')

        self.airline_encoder = LabelEncoder()
        flights_delays.loc[:, "AIRLINE_ENCODED"] = self.airline_encoder.fit_transform(flights_delays["AIRLINE"])

        #flights_delays = pd.get_dummies(flights_delays, columns=["AIRLINE"], dtype=int)

        print('Variáveis das companhias aéreas criado com sucesso')

        print('Criando Clusters...')

        route_stats = flights_delays.groupby("ROUTE").agg({
            "DEPARTURE_DELAY": ["mean", "std", "count"],
            "DISTANCE": "mean"
        }).reset_index()

        route_stats.columns = ["ROUTE", "DELAY_MEAN", "DELAY_STD", "COUNT", "DISTANCE"]
        route_stats = route_stats.fillna(0)

        route_stats = self.cluster(route_stats)
        flights_delays = flights_delays.merge(route_stats[["ROUTE", "ROUTE_CLUSTER"]], on="ROUTE", how="left")

        print('Clusters criados')

        print('Dados para análise de atrasos criados com sucesso')

        drop_cols = ['AIRLINE','MONTH','DAY','DAY_OF_WEEK','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY','HAS_WEATHER_DELAY','HAS_AIRLINE_DELAY','HAS_AIR_SYSTEM_DELAY','HAS_SECURITY_DELAY','HAS_LATE_AIRCRAFT_DELAY','DEP_MINUTES','SEASON', 'ROUTE']
        data_to_classifier_model = flights_delays.drop(drop_cols, axis=1)

        drop_cols=['HAD_DELAY','AIRLINE','MONTH','DAY','DAY_OF_WEEK','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY','DEP_MINUTES','SEASON', 'ROUTE']
        delay_data = flights_delays[flights_delays["HAD_DELAY"] == 1]
        data_to_regression_model = delay_data[delay_data["DEPARTURE_DELAY"] < 87]
        data_to_regression_model = data_to_regression_model.drop(drop_cols, axis=1)

        print("============================================")

        total_data = data_to_classifier_model.shape[0]
        data_has_delay = delay_data.shape[0]
        data_hasnt_delay = total_data - data_has_delay

        print(f'Dados totais = {total_data}')
        print(f'Dados que tiveram atraso = {data_has_delay}')
        print(f'Dados que não tiveram atraso = {data_hasnt_delay}')
        print("============================================")

        self._classifier_model = self.classifier_model(data_to_classifier_model)
        self._regression_model = self.regression_model(data_to_regression_model)

    async def analysis_do_task(self):
        if self.RUNNING:
            return
        async with self.lock:
            loop = asyncio.get_running_loop()
            if self.executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self.executor = ThreadPoolExecutor(max_workers=2)

            self.RUNNING = True
            await loop.run_in_executor(self.executor, self.analysis_task)
            self.RUNNING = False

    async def analysis(self):
        asyncio.create_task(self.analysis_do_task())

    async def regression_predict(self, data: FlightRegression):
        if self.airports is None:
            return None
        
        predict_data = data.to_dataframe()
        lookup = self.airports.set_index("IATA_CODE")

        predict_data.loc[:,"ORIGIN_LAT"] = predict_data["ORIGIN"].map(lookup["LATITUDE"])
        predict_data.loc[:,"ORIGIN_LON"] = predict_data["ORIGIN"].map(lookup["LONGITUDE"])

        predict_data.loc[:,"DESTINATION_LAT"] = predict_data["DESTINATION"].map(lookup["LATITUDE"])
        predict_data.loc[:,"DESTINATION_LON"] = predict_data["DESTINATION"].map(lookup["LONGITUDE"])

        predict_data.loc[:,"DELTA_LAT"] = predict_data["DESTINATION_LAT"] - predict_data["ORIGIN_LAT"]
        predict_data.loc[:,"DELTA_LON"] = predict_data["DESTINATION_LON"] - predict_data["ORIGIN_LON"]

        lat1 = np.radians(predict_data["ORIGIN_LAT"])
        lon1 = np.radians(predict_data["ORIGIN_LON"])
        lat2 = np.radians(predict_data["DESTINATION_LAT"])
        lon2 = np.radians(predict_data["DESTINATION_LON"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        predict_data.loc[:,"DISTANCE_HAVERSINE"] = 6371 * c

        predict_data.loc[:, "AIRLINE_ENCODED"] = self.airline_encoder.transform(predict_data["AIRLINE"])
        predict_data.loc[:, "SEASON_ENCODED"] = self.season_encoder.transform(predict_data["SEASON"])

        predict_data.loc[:, "ROUTE"] = predict_data["ORIGIN"] + "_" + predict_data["DESTINATION"]

        predict_data = predict_data.merge(
            self.route_stats,
            on="ROUTE",
            how="left"
        )

        predict_data = predict_data.fillna(0)

        drop_cols=['AIRLINE','SEASON', 'ROUTE', 'ORIGIN', 'DESTINATION', "DELAY_MEAN", "DELAY_STD", "COUNT"]
        predict_data = predict_data.drop(drop_cols, axis=1)

        predict_data = predict_data[self.regression_feature_columns]
        predict_value = self._regression_model.predict(predict_data)
        predict_value = np.expm1(predict_value)

        return float(predict_value[0])
    
    async def classifier_predict(self, data: FlightClassifier):
        if self.airports is None:
            return None
        
        predict_data = data.to_dataframe()
        lookup = self.airports.set_index("IATA_CODE")

        predict_data.loc[:,"ORIGIN_LAT"] = predict_data["ORIGIN"].map(lookup["LATITUDE"])
        predict_data.loc[:,"ORIGIN_LON"] = predict_data["ORIGIN"].map(lookup["LONGITUDE"])

        predict_data.loc[:,"DESTINATION_LAT"] = predict_data["DESTINATION"].map(lookup["LATITUDE"])
        predict_data.loc[:,"DESTINATION_LON"] = predict_data["DESTINATION"].map(lookup["LONGITUDE"])

        predict_data.loc[:,"DELTA_LAT"] = predict_data["DESTINATION_LAT"] - predict_data["ORIGIN_LAT"]
        predict_data.loc[:,"DELTA_LON"] = predict_data["DESTINATION_LON"] - predict_data["ORIGIN_LON"]

        lat1 = np.radians(predict_data["ORIGIN_LAT"])
        lon1 = np.radians(predict_data["ORIGIN_LON"])
        lat2 = np.radians(predict_data["DESTINATION_LAT"])
        lon2 = np.radians(predict_data["DESTINATION_LON"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        predict_data.loc[:,"DISTANCE_HAVERSINE"] = 6371 * c

        predict_data.loc[:, "AIRLINE_ENCODED"] = self.airline_encoder.transform(predict_data["AIRLINE"])
        predict_data.loc[:, "SEASON_ENCODED"] = self.season_encoder.transform(predict_data["SEASON"])

        predict_data.loc[:, "ROUTE"] = predict_data["ORIGIN"] + "_" + predict_data["DESTINATION"]

        predict_data.loc[:, "ROUTE"] = predict_data["ORIGIN"] + "_" + predict_data["DESTINATION"]

        predict_data = predict_data.merge(
            self.route_stats,
            on="ROUTE",
            how="left"
        )

        predict_data = predict_data.fillna(0)

        drop_cols=['AIRLINE','SEASON', 'ROUTE', 'ORIGIN', 'DESTINATION', "DELAY_MEAN", "DELAY_STD", "COUNT"]
        predict_data = predict_data.drop(drop_cols, axis=1)

        predict_data = predict_data[self.classifier_feature_columns]
        predict_value = self._classifier_model.predict(predict_data)
        
        return int(predict_value[0])