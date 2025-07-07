import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar errores de tkinter
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import json
import os
import joblib
import io
import base64
from threading import Lock
from functools import wraps
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Crear un lock para manejar la concurrencia en matplotlib
matplotlib_lock = Lock()

app = Flask(__name__)
CORS(app)  # Restringir orígenes para CORS

# Decorador para timeout
def timeout(seconds):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            if time.time() - start > seconds:
                abort(504, description="Request timed out")
            return result
        return wrapper
    return decorator

def load_and_preprocess_data(file_path="Datos_Completos.csv"):
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file does not exist at path: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Attempting to load CSV from: {os.path.abspath(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(5)]
            logger.info(f"First 5 lines of CSV:\n{''.join(first_lines)}")

        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.lower() == 'datos_completos':
                logger.info("Detected title row 'Datos_Completos' in first line, skipping it")
                skiprows = 1
            else:
                skiprows = 0

        df = pd.read_csv(file_path, skipinitialspace=True, encoding='utf-8', skiprows=skiprows)
        df.columns = df.columns.str.strip()

        expected_columns = [
            'age', 'gender', 'Academic_Level', 'Country', 'avg_daily_usage_hours',
            'Most_Used_Platform', 'affects_academic_performance', 'sleep_hours_per_night',
            'mental_health_score', 'relationship_status_single', 'conflicts_over_social_media',
            'addicted_score'
        ]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Faltan columnas en el CSV: {missing_columns}")

        numeric_columns = ['age', 'avg_daily_usage_hours', 'sleep_hours_per_night', 
                          'affects_academic_performance', 'mental_health_score', 
                          'conflicts_over_social_media', 'addicted_score']
        for col in numeric_columns:
            df[col] = df[col].replace('-', pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"Converted {col} to numeric, NaN count: {df[col].isna().sum()}")

        df = df[(df['age'] >= 10) & (df['age'] <= 80)]
        df = df[(df['avg_daily_usage_hours'] >= 0) & (df['avg_daily_usage_hours'] <= 24)]
        df = df[(df['sleep_hours_per_night'] >= 0) & (df['sleep_hours_per_night'] <= 16)]

        categorical_features = ['gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 
                              'relationship_status_single']
        for col in categorical_features:
            df[col] = df[col].str.strip().str.capitalize()

        df['gender'] = df['gender'].replace({'F': 'Female', 'Otro': 'Other'})
        df['relationship_status_single'] = df['relationship_status_single'].replace({
            'Complicado': 'Complicated', 'In a relationshipn': 'In a relationship', '3': 'Unknown'})
        df['Country'] = df['Country'].replace({
            'México': 'Mexico', 'Estados unidos': 'Usa', 'Brasil': 'Brazil'})

        df['addicted_score'] = df.apply(
            lambda x: (4 if x['avg_daily_usage_hours'] > 5 else 0) +
                      (3 if x['affects_academic_performance'] == 1 else 0) +
                      (3 if x['sleep_hours_per_night'] < 6 else 0), axis=1)

        df['conflicts_over_social_media'] = df.apply(
            lambda x: 1 if x['avg_daily_usage_hours'] > 5 and 
                          x['relationship_status_single'] in ['Single', 'Complicated'] else 0, axis=1)

        df['high_social_media_usage'] = (df['avg_daily_usage_hours'] > 5).astype(int)
        df['low_sleep_quality'] = (df['sleep_hours_per_night'] < 6).astype(int)

        category_map = {col: df[col].unique().tolist() for col in categorical_features}
        with open('category_map.json', 'w') as f:
            json.dump(category_map, f)

        for col in categorical_features:
            logger.info(f"Categorías únicas para {col}: {category_map[col]}")

        logger.info(f"Loaded and preprocessed {len(df)} rows from {file_path}")
        return df

    except Exception as e:
        logger.error(f"Error al cargar el CSV: {e}")
        raise

base_features = [
    'age', 'avg_daily_usage_hours', 'gender', 'Academic_Level',
    'Country', 'Most_Used_Platform', 'affects_academic_performance',
    'relationship_status_single', 'mental_health_score', 'sleep_hours_per_night'
]

numerical_features = ['age', 'avg_daily_usage_hours', 'mental_health_score', 
                     'affects_academic_performance', 'sleep_hours_per_night']
categorical_features = ['gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 
                      'relationship_status_single']

regression_features = {
    'sleep_hours_per_night': [
        'age', 'avg_daily_usage_hours', 'gender', 'Academic_Level', 'Country',
        'Most_Used_Platform', 'affects_academic_performance', 'relationship_status_single',
        'mental_health_score'
    ],
    'avg_daily_usage_hours': [
        'age', 'gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
        'affects_academic_performance', 'relationship_status_single', 'sleep_hours_per_night', 'mental_health_score'
    ],
    'addicted_score': [
        'age', 'avg_daily_usage_hours', 'gender', 'Academic_Level', 'Country',
        'Most_Used_Platform', 'affects_academic_performance', 'relationship_status_single',
        'mental_health_score', 'sleep_hours_per_night'
    ]
}

classification_features = {
    'affects_academic_performance': base_features,
    'conflicts_over_social_media': base_features,
    'high_social_media_usage': base_features,
    'low_sleep_quality': base_features
}

def train_models(df):
    regression_targets = {
        'sleep_hours_per_night': df['sleep_hours_per_night'],
        'avg_daily_usage_hours': df['avg_daily_usage_hours'],
        'addicted_score': df['addicted_score']
    }

    classification_targets = {
        'affects_academic_performance': df['affects_academic_performance'],
        'conflicts_over_social_media': df['conflicts_over_social_media'],
        'high_social_media_usage': df['high_social_media_usage'],
        'low_sleep_quality': df['low_sleep_quality']
    }

    regression_models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest_Regression': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision_Tree_Regression': DecisionTreeRegressor(random_state=42)
    }

    classification_models = {
        'Logistic_Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
        'Decision_Tree_Classifier': DecisionTreeClassifier(random_state=42)
    }

    try:
        with open('category_map.json', 'r') as f:
            category_map = json.load(f)
    except FileNotFoundError:
        logger.error("category_map.json no encontrado")
        raise
    except json.JSONDecodeError:
        logger.error("Error al decodificar category_map.json")
        raise

    regression_pipelines = {}
    for target_name, y in regression_targets.items():
        X = df[regression_features[target_name]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        regression_pipelines[target_name] = {}
        local_num = [col for col in numerical_features if col in X.columns]
        local_cat = [col for col in categorical_features if col in X.columns]
        categories = [category_map[col] for col in local_cat]
        local_preprocessor = ColumnTransformer([
            ('num', StandardScaler(), local_num),
            ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False), local_cat)
        ])
        for model_name, model in regression_models.items():
            pipeline = Pipeline([('preprocessor', local_preprocessor), ('model', model)])
            logger.info(f"Training {model_name} for {target_name} with features: {X.columns.tolist()}")
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            logger.info(f"{model_name} for {target_name} MSE: {mse}")
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            logger.info(f"Features for {target_name} - {model_name}: {len(feature_names)} features")
            logger.info(f"Feature names: {list(feature_names)}")
            regression_pipelines[target_name][model_name] = pipeline
            joblib.dump(pipeline, f"models/{target_name}_{model_name}.joblib")
            logger.info(f"Saved pipeline for {target_name} - {model_name} to models/{target_name}_{model_name}.joblib")

    classification_pipelines = {}
    for target_name, y in classification_targets.items():
        X = df[classification_features[target_name]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classification_pipelines[target_name] = {}
        local_num = [col for col in numerical_features if col in X.columns]
        local_cat = [col for col in categorical_features if col in X.columns]
        categories = [category_map[col] for col in local_cat]
        local_preprocessor = ColumnTransformer([
            ('num', StandardScaler(), local_num),
            ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False), local_cat)
        ])
        for model_name, model in classification_models.items():
            pipeline = Pipeline([('preprocessor', local_preprocessor), ('model', model)])
            logger.info(f"Training {model_name} for {target_name} with features: {X.columns.tolist()}")
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, pred)
            logger.info(f"{model_name} for {target_name} Accuracy: {acc}")
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            logger.info(f"Features for {target_name} - {model_name}: {len(feature_names)} features")
            logger.info(f"Feature names: {list(feature_names)}")
            classification_pipelines[target_name][model_name] = pipeline
            joblib.dump(pipeline, f"models/{target_name}_{model_name}.joblib")
            logger.info(f"Saved pipeline for {target_name} - {model_name} to models/{target_name}_{model_name}.joblib")

    kmeans_features = ['age', 'avg_daily_usage_hours', 'sleep_hours_per_night']
    X_kmeans = df[kmeans_features]
    scaler = StandardScaler()
    X_kmeans_scaled = scaler.fit_transform(X_kmeans)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_kmeans_scaled)
    joblib.dump(kmeans, "models/kmeans.joblib")
    joblib.dump(scaler, "models/kmeans_scaler.joblib")

    cluster_descriptions = []
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, center in enumerate(cluster_centers):
        desc = f"Cluster {i}: Usuarios con edad promedio {center[0]:.1f}, uso diario de {center[1]:.1f} horas, durmiendo {center[2]:.1f} horas por noche."
        cluster_data = df[kmeans.labels_ == i]
        common_gender = cluster_data['gender'].mode()[0] if not cluster_data['gender'].empty else 'Desconocido'
        common_platform = cluster_data['Most_Used_Platform'].mode()[0] if not cluster_data['Most_Used_Platform'].empty else 'Desconocido'
        common_academic = cluster_data['Academic_Level'].mode()[0] if not cluster_data['Academic_Level'].empty else 'Desconocido'
        desc += f" Predominan usuarios de género {common_gender}, usando principalmente {common_platform}, y con nivel académico {common_academic}."
        cluster_descriptions.append(desc)

    return regression_pipelines, classification_pipelines, kmeans, scaler, cluster_descriptions

if not os.path.exists("models"):
    os.makedirs("models")

try:
    regression_pipelines = {}
    classification_pipelines = {}
    for target in ['sleep_hours_per_night', 'avg_daily_usage_hours', 'addicted_score']:
        regression_pipelines[target] = {}
        for model in ['Linear_Regression', 'Random_Forest_Regression', 'Decision_Tree_Regression']:
            model_path = f"models/{target}_{model}.joblib"
            if os.path.exists(model_path):
                regression_pipelines[target][model] = joblib.load(model_path)
                logger.info(f"Loaded pipeline for {target} - {model} from {model_path}")
            else:
                logger.info(f"Model file {model_path} not found, will train new models")
                raise FileNotFoundError
    for target in ['affects_academic_performance', 'conflicts_over_social_media', 'high_social_media_usage', 'low_sleep_quality']:
        classification_pipelines[target] = {}
        for model in ['Logistic_Regression', 'Decision_Tree_Classifier']:
            model_path = f"models/{target}_{model}.joblib"
            if os.path.exists(model_path):
                classification_pipelines[target][model] = joblib.load(model_path)
                logger.info(f"Loaded pipeline for {target} - {model} from {model_path}")
            else:
                logger.info(f"Model file {model_path} not found, will train new models")
                raise FileNotFoundError
    kmeans = joblib.load("models/kmeans.joblib")
    kmeans_scaler = joblib.load("models/kmeans_scaler.joblib")
    df = load_and_preprocess_data()
    _, _, _, _, cluster_descriptions = train_models(df)
except FileNotFoundError:
    df = load_and_preprocess_data()
    regression_pipelines, classification_pipelines, kmeans, kmeans_scaler, cluster_descriptions = train_models(df)

@app.route('/predict', methods=['POST'])
@timeout(30)  # 30 segundos de timeout
def predict():
    start_time = time.time()
    logger.info("Iniciando procesamiento de la solicitud")
    try:
        data = request.get_json()
        logger.info(f"Received input data: {data}")
        required_fields = [
            'age', 'gender', 'Academic_Level', 'Country', 'avg_daily_usage_hours',
            'Most_Used_Platform', 'affects_academic_performance', 'sleep_hours_per_night',
            'relationship_status_single'
        ]
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            logger.error(f"Faltan campos requeridos: {missing}")
            return jsonify({'error': f'Faltan campos requeridos: {missing}'}), 400

        numeric_fields = ['age', 'avg_daily_usage_hours', 'sleep_hours_per_night', 'affects_academic_performance']
        for field in numeric_fields:
            try:
                data[field] = float(data[field])
                logger.info(f"Validated {field} as numeric: {data[field]}")
            except (ValueError, TypeError):
                logger.error(f"Valor no numérico para {field}: {data[field]}")
                return jsonify({'error': f"Valor no numérico para {field}: {data[field]}"}), 400

        # Validaciones adicionales para rangos
        if not (10 <= data['age'] <= 80):
            logger.error(f"Edad fuera de rango: {data['age']}")
            return jsonify({'error': 'Edad debe estar entre 10 y 80'}), 400
        if not (0 <= data['avg_daily_usage_hours'] <= 24):
            logger.error(f"Horas de uso diario fuera de rango: {data['avg_daily_usage_hours']}")
            return jsonify({'error': 'Horas de uso diario deben estar entre 0 y 24'}), 400
        if not (0 <= data['sleep_hours_per_night'] <= 16):
            logger.error(f"Horas de sueño fuera de rango: {data['sleep_hours_per_night']}")
            return jsonify({'error': 'Horas de sueño deben estar entre 0 y 16'}), 400

        try:
            with open('category_map.json', 'r') as f:
                category_map = json.load(f)
        except FileNotFoundError:
            logger.error("category_map.json no encontrado")
            return jsonify({'error': 'Archivo category_map.json no encontrado'}), 500
        except json.JSONDecodeError:
            logger.error("Error al decodificar category_map.json")
            return jsonify({'error': 'Error al decodificar category_map.json'}), 500

        for col in categorical_features:
            data[col] = data[col].capitalize()
            if data[col] not in category_map[col]:
                logger.warning(f"Valor {data[col]} para {col} no visto en entrenamiento, usando primera categoría por defecto")
                data[col] = category_map[col][0]

        data['mental_health_score'] = df['mental_health_score'].median()
        logger.info(f"Imputed mental_health_score: {data['mental_health_score']}")

        input_data = pd.DataFrame([data], columns=base_features)
        logger.info(f"Input DataFrame columns: {input_data.columns.tolist()}")
        logger.info(f"Input DataFrame values: {input_data.to_dict(orient='records')}")

        numeric_columns = ['age', 'avg_daily_usage_hours', 'sleep_hours_per_night', 
                          'affects_academic_performance', 'mental_health_score']
        for col in numeric_columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
            if input_data[col].isna().any():
                logger.error(f"Valor numérico inválido para {col}: {input_data[col].iloc[0]}")
                return jsonify({'error': f"Valor numérico inválido para {col}: {input_data[col].iloc[0]}"}), 400

        dataset_stats = {
            'sleep_hours_per_night': df['sleep_hours_per_night'].median(),
            'avg_daily_usage_hours': df['avg_daily_usage_hours'].mean(),
            'addicted_score': 7  
        }

        regression_predictions = {}
        regression_charts = {}
        regression_chart_descriptions = {
            'sleep_hours_per_night': 'Este gráfico de líneas muestra las predicciones de horas de sueño por noche para tres modelos. La línea horizontal representa la mediana de horas de sueño del conjunto de datos (referencia poblacional). El punto rojo indica tu valor predicho promedio. Un valor por encima de la mediana sugiere mejor calidad de sueño que el promedio.',
            'avg_daily_usage_hours': 'Este gráfico de barras muestra las predicciones de horas de uso diario de redes sociales para tres modelos. La línea horizontal representa la media de uso diario del conjunto de datos. La estrella verde marca tu valor predicho promedio. Un valor por encima de la media indica un uso más intensivo que el promedio.',
            'addicted_score': 'Este gráfico de dispersión muestra las predicciones de puntaje de adicción (0-10) para tres modelos. La línea horizontal en 7 indica el umbral de alto riesgo de dependencia. El diamante azul marca tu valor predicho promedio. Un valor por encima de 7 sugiere un riesgo elevado de adicción a las redes sociales.'
        }

        for target_name, models in regression_pipelines.items():
            features = regression_features[target_name]
            input_data_subset = input_data[features].copy()
            regression_predictions[target_name] = {}
            model_names = []
            predictions = []
            for name, pipeline in models.items():
                transformed_input = pipeline.named_steps['preprocessor'].transform(input_data_subset)
                prediction = pipeline.predict(input_data_subset)[0]
                regression_predictions[target_name][name] = float(prediction)
                model_names.append(name.replace('_', ' '))
                predictions.append(prediction)

            avg_prediction = np.mean(predictions)

            with matplotlib_lock:
                plt.figure(figsize=(6, 4))  # Tamaño más pequeño para optimizar
                if target_name == 'sleep_hours_per_night':
                    plt.plot(model_names, predictions, marker='o', color='#4C78A8', linewidth=2, markersize=8, label='Predicciones')
                    plt.axhline(y=dataset_stats['sleep_hours_per_night'], color='gray', linestyle='--', label=f'Mediana del dataset ({dataset_stats["sleep_hours_per_night"]:.1f} horas)')
                    plt.plot(['Promedio'], [avg_prediction], marker='o', color='red', markersize=12, label='Tu valor promedio')
                    plt.title('Predicciones de Horas de Sueño por Noche')
                    plt.ylabel('Horas')
                    plt.ylim(0, 16)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                elif target_name == 'avg_daily_usage_hours':
                    bars = plt.bar(model_names, predictions, color='#F58518')
                    plt.axhline(y=dataset_stats['avg_daily_usage_hours'], color='gray', linestyle='--', label=f'Media del dataset ({dataset_stats["avg_daily_usage_hours"]:.1f} horas)')
                    plt.plot(['Promedio'], [avg_prediction], marker='*', color='green', markersize=15, label='Tu valor promedio')
                    plt.title('Predicciones de Horas de Uso Diario de Redes')
                    plt.ylabel('Horas')
                    plt.ylim(0, 24)
                    plt.legend()
                    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{height:.1f}', ha='center', va='bottom')
                else:
                    plt.scatter(model_names, predictions, color='#E45756', s=100, label='Predicciones')
                    plt.axhline(y=dataset_stats['addicted_score'], color='gray', linestyle='--', label='Umbral de alto riesgo (7)')
                    plt.scatter(['Promedio'], [avg_prediction], color='blue', marker='D', s=150, label='Tu valor promedio')
                    plt.title('Predicciones de Puntaje de Adicción')
                    plt.ylabel('Puntaje (0-10)')
                    plt.ylim(0, 10)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                regression_charts[target_name] = {
                    'image': f'data:image/png;base64,{chart_base64}',
                    'description': regression_chart_descriptions[target_name]
                }
                plt.close()

        classification_predictions = {}
        for target_name, models in classification_pipelines.items():
            input_data_subset = input_data[classification_features[target_name]]
            classification_predictions[target_name] = {}
            for name, pipeline in models.items():
                transformed_input = pipeline.named_steps['preprocessor'].transform(input_data_subset)
                if name == 'Logistic_Regression':
                    probas = pipeline.named_steps['model'].predict_proba(transformed_input)[0]
                    prediction = pipeline.predict(input_data_subset)[0]
                    classification_predictions[target_name][name] = {
                        'prediction': int(prediction),
                        'confidence': float(probas[prediction])
                    }
                else:
                    prediction = pipeline.predict(input_data_subset)[0]
                    classification_predictions[target_name][name] = {'prediction': int(prediction)}

        kmeans_input = pd.DataFrame({
            'age': [data['age']],
            'avg_daily_usage_hours': [data['avg_daily_usage_hours']],
            'sleep_hours_per_night': [data['sleep_hours_per_night']]
        }, columns=['age', 'avg_daily_usage_hours', 'sleep_hours_per_night'])
        kmeans_input_scaled = kmeans_scaler.transform(kmeans_input)
        cluster = int(kmeans.predict(kmeans_input_scaled)[0])

        addicted_score = regression_predictions['addicted_score']['Random_Forest_Regression']
        usage_hours = regression_predictions['avg_daily_usage_hours']['Random_Forest_Regression']
        low_sleep = classification_predictions['low_sleep_quality']['Logistic_Regression']['prediction']
        dependency_risk = min(100, (addicted_score / 10 * 50) + (usage_hours / 24 * 30) + (low_sleep * 20))

        response = {
            'Regression_Predictions': regression_predictions,
            'Regression_Charts': regression_charts,
            'Classification_Predictions': classification_predictions,
            'KMeans_Cluster': cluster,
            'KMeans_Cluster_Description': cluster_descriptions[cluster],
            'Dependency_Risk_Score': dependency_risk
        }

        logger.info(f"Procesamiento completado en {time.time() - start_time:.2f} segundos")
        logger.info(f"Final response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error de predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=8000)