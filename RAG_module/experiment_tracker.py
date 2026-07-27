import json
import os
import pandas as pd
from datetime import datetime


TRACKER_PATH = "RAG_module/experiments.json"

def load_experiments():
  if not os.path.exists(TRACKER_PATH):
    return []
  with open(TRACKER_PATH, "r") as f:
    return json.load(f)

def save_experiment(params, metrics, notes=""):
  experiments = load_experiments()
  record = {
      "id": len(experiments) + 1,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
      "params": params,
      "metrics": metrics,
      "notes": notes
  }
  experiments.append(record)
  with open(TRACKER_PATH, "w") as f:
    json.dump(experiments, f, indent=2)

  print(f"Experiment #{record['id']} saved.")

def print_experiments():
  experiments = load_experiments()
  for exp in experiments:
    print(f"\n#{exp['id']} | {exp['timestamp']}")
    print(f"  params : {exp['params']}")
    print(f"  metrics: {exp['metrics']}")
    if exp['notes']:
      print(f"  notes  : {exp['notes']}")

def show_experiments():
  experiments = load_experiments()
  if not experiments:
    print("no experiments yet")
    return

  rows = []
  for exp in experiments:
    row = {"id": exp["id"], "timestamp": exp["timestamp"]}
    row.update(exp["params"])
    row.update(exp["metrics"])
    row["notes"] = exp["notes"]
    rows.append(row)

  df = pd.DataFrame(rows)
  return df