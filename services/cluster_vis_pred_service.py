import pandas as pd
import os
import json
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from ast import literal_eval
import pathlib

from configs.llm_model import embeddings, model, client
from scripts.promting import PROMPT_REPORT

class ChallengeClusterService:
    def __init__(self) -> None:
        pass

    def call_llm(self, prompt) -> str:
        completion = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                      ],
            max_tokens=1000,
        )
        message_openai = completion.choices[0].message.content.lstrip("\n")
        message_openai = message_openai.replace("\n", "")
        return message_openai
    
    def visuallize(self, data, num_clusters=4, synthetic_data=True):
        if synthetic_data == False:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            # create df[text] = df["gender"] + df["age"] + df["career"] + df["interest"]
            df["fulltext"] = df["gender"] + " " + df["age"] + " " + df["career"] + " " + df["interest"]
            df['embeddings'] = df["fulltext"].apply(lambda x : embeddings.embed_query(x))
            df.to_csv("data/embeddings.csv", index=False)
        else:
            df = pd.read_csv("data/embeddings.csv")
        
        df["embedding"] = df.embeddings.apply(literal_eval).apply(np.array)
        df.drop(columns=["embeddings"], inplace=True)
        matrix = np.vstack(df.embedding.values)

        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        df["Cluster"] = labels

        group_fd = df.groupby("Cluster")

        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        vis_dims2 = tsne.fit_transform(matrix)

        df['x'] = vis_dims2[:, 0]
        df['y'] = vis_dims2[:, 1]

        # Preparing JSON output
        json_output = {"datasets": []}

        for cluster, group in group_fd:
            cluster_data = {
                "label": f"Cluster {cluster + 1}",
                "data": group[['x', 'y']].to_dict(orient='records')
            }
            json_output["datasets"].append(cluster_data)

        return json_output
    
    def export_report(self, data, synthetic_data = True):
        str_data = json.dumps(data, indent=4)
        prompt = PROMPT_REPORT.replace("{data_info}", str_data)
        print("getting report")
        bot_answer = self.call_llm(prompt)
        return bot_answer
