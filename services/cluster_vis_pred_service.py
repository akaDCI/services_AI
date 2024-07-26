import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

class ChallengeClusterService:
    def __init__(self) -> None:
        pass

    def visuallize(self, data, num_clusters=4):
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Encode categorical variables
        le_gender = LabelEncoder()
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])

        le_career = LabelEncoder()
        df['career_encoded'] = le_career.fit_transform(df['career'])

        le_interest = LabelEncoder()
        df['interest_encoded'] = le_interest.fit_transform(df['interest'])

        # Standardize age
        scaler = StandardScaler()
        df['age_scaled'] = scaler.fit_transform(df[['age']])

        # Prepare features for clustering
        features = df[['gender_encoded', 'age_scaled', 'career_encoded', 'interest_encoded']]

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(features)

        # 3D Visualization
        fig = px.scatter_3d(df, x='age', y='interest', z='career', color='cluster', 
                            hover_data=['gender', 'age', 'career', 'interest'], 
                            title="3D Clustering Visualization")

        fig.show()