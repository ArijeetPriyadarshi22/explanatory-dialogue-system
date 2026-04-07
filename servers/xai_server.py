import os
import json
import pandas as pd
import numpy as np
import shap
import dice_ml
from dice_ml import Dice
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fastmcp import FastMCP

# ====== Data & Model bootstrap (Titanic demo) ======

def bootstrap_model():
    titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    titanic = titanic[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    titanic["Sex"] = LabelEncoder().fit_transform(titanic["Sex"])  # Male=1, Female=0 (sklearn default)
    X = titanic[["Pclass", "Sex", "Age", "Fare"]]
    y = titanic["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    data_meta = {
        "feature_names": list(X.columns),
        "categorical": ["Sex"],
        "continuous": ["Pclass", "Age", "Fare"]
    }

    dice_data = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=data_meta["continuous"] + ["Sex"],  # treat as continuous for simple demo
        outcome_name="Survived",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice = Dice(dice_data, dice_model, method="random")

    return model, explainer, dice, data_meta

model, explainer, dice, data_meta = bootstrap_model()

# ====== MCP servers definition ======
server = FastMCP("xai-servers")

@server.tool()
def predict(instance: dict) -> dict:
    import pandas as pd
    df = pd.DataFrame([instance])
    # align columns
    df = df.reindex(columns=data_meta["feature_names"], fill_value=np.nan)
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][pred])
    return {"prediction": pred, "probability": proba}

@server.tool()
def feature_info() -> dict:
    return data_meta

@server.tool()
def shap_explain(instance: dict) -> dict:
    import pandas as pd
    df = pd.DataFrame([instance])
    df = df.reindex(columns=data_meta["feature_names"], fill_value=np.nan)
    shap_vals = explainer.shap_values(df)
    pred = int(model.predict(df)[0])
    values = shap_vals[pred][0].tolist() if isinstance(shap_vals, list) else shap_vals[0].tolist()
    return {
        "predicted_class": pred,
        "feature_names": data_meta["feature_names"],
        "shap_values": values,
        "base_value": float(explainer.expected_value[pred] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
    }

@server.tool()
def shap_top_features(instance: dict, k: int = 3) -> dict:
    res = shap_explain(instance)
    vals = np.abs(np.array(res["shap_values"]))
    idx = np.argsort(vals)[::-1][:k]
    tops = [
        {"feature": res["feature_names"][i], "abs_shap": float(vals[i])}
        for i in idx
    ]
    return {"predicted_class": res["predicted_class"], "top": tops}

@server.tool()
def generate_cfs(instance: dict, total_cfs: int = 3, desired_class: str = "opposite", features_to_vary: list | None = None) -> dict:
    import pandas as pd
    df = pd.DataFrame([instance])
    exp = dice.generate_counterfactuals(df, total_CFs=total_cfs, desired_class=desired_class, features_to_vary=features_to_vary)
    out = []
    for ex in exp.cf_examples_list:
        cf_df = ex.final_cfs_df
        for _, row in cf_df.iterrows():
            out.append({k: (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v)) for k, v in row.to_dict().items()})
    return {"counterfactuals": out}

if __name__ == "__main__":
    # Runs an MCP stdio servers
    server.run(transport="stdio")
