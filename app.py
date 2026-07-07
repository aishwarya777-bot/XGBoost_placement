import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Updated to match the exact filename in your GitHub repository
MODEL_PATH = "xgboost_placement.pkl"
model = None
model_error = None

# Wrap the pickle loading in a try-except block so the app doesn't crash on startup
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        model_error = f"File not found: {MODEL_PATH}. Make sure your model is named 'xgboost_placement.pkl' and is in the root folder."
        print(f"❌ {model_error}")
except Exception as e:
    model_error = f"Pickle loading failed. This usually means a version mismatch. Error: {str(e)}"
    print(f"❌ {model_error}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placement Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    </style>
</head>
<body class="min-h-screen text-slate-100 flex items-center justify-center p-4 antialiased">

    <div class="w-full max-w-4xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl overflow-hidden grid md:grid-cols-5">
        
        <!-- Left Side Branding Pane -->
        <div class="md:col-span-2 bg-gradient-to-br from-indigo-600 to-violet-600 p-8 flex flex-col justify-between text-white">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight mb-2">CampusAI</h1>
                <p class="text-indigo-100 text-sm leading-relaxed">
                    Evaluate students' real-time probability of recruitment success using key predictive academic performance indicators.
                </p>
            </div>
            <div class="mt-8 md:mt-0 pt-6 border-t border-indigo-500/30">
                <span class="text-xs uppercase tracking-wider text-indigo-200 font-semibold block mb-1">Architecture</span>
                <span class="text-sm font-mono bg-indigo-900/40 px-2 py-1 rounded border border-indigo-500/20 inline-block">XGBoost Classifier</span>
            </div>
        </div>

        <!-- Right Side Form Pane -->
        <div class="md:col-span-3 p-8 bg-slate-900/40">
            <h2 class="text-xl font-bold mb-6 text-slate-200">Student Profile Metrics</h2>
            
            {% if model_error %}
            <div class="bg-rose-500/20 border border-rose-500/40 text-rose-300 p-4 rounded-xl mb-6 text-sm">
                <strong>Model Configuration Error:</strong> {{ model_error }}
            </div>
            {% endif %}

            <form method="POST" action="/" class="space-y-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Study Hours</label>
                        <input type="number" name="study_hours" required min="0" max="168" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 15">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Attendance (%)</label>
                        <input type="number" name="attendance" required min="0" max="100" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 95">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Daily Sleep Hours</label>
                        <input type="number" name="sleep_hours" required min="0" max="24" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 7">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Internet Usage</label>
                        <input type="number" name="internet_usage" required min="0" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="Hours or Index">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Assignments Completed</label>
                        <input type="number" name="assignments_completed" required min="0" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 10">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Previous Score</label>
                        <input type="number" name="previous_score" required min="0" max="100" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 78">
                    </div>
                </div>
                
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Current Exam Score</label>
                    <input type="number" step="0.01" name="exam_score" required min="0" max="100" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-indigo-500 transition" placeholder="e.g. 84.50">
                </div>

                <button type="submit" {% if model_error %}disabled class="w-full mt-2 bg-slate-700 cursor-not-allowed font-bold py-3 px-4 rounded-lg text-slate-400"{% else %}class="w-full mt-2 bg-gradient-to-r from-indigo-500 to-violet-500 hover:from-indigo-600 hover:to-violet-600 font-bold py-3 px-4 rounded-lg shadow-lg transform active:scale-[0.98] transition"{% endif %}>
                    Predict Placement Outcome
                </button>
            </form>

            {% if status is not none %}
            <div class="mt-6 pt-6 border-t border-slate-800">
                <div class="p-4 rounded-xl {% if status == 'Placed' %}bg-emerald-500/10 border border-emerald-500/30{% else %}bg-rose-500/10 border border-rose-500/30{% endif %} flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div>
                        <p class="text-xs font-bold uppercase tracking-widest text-slate-400">Analysis Status</p>
                        <h3 class="text-2xl font-black {% if status == 'Placed' %}text-emerald-400{% else %}text-rose-400{% endif %}">
                            {{ status | upper }}
                        </h3>
                    </div>
                    <div class="text-right">
                        <span class="text-xs text-slate-400 block">Confidence Metric</span>
                        <span class="text-xl font-mono font-bold text-slate-200">{{ confidence }}%</span>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    status = None
    confidence = None
    
    if request.method == "POST" and model is not None:
        features = [
            int(request.form["study_hours"]),
            int(request.form["attendance"]),
            int(request.form["sleep_hours"]),
            int(request.form["internet_usage"]),
            int(request.form["assignments_completed"]),
            int(request.form["previous_score"]),
            float(request.form["exam_score"])
        ]
        
        df_input = pd.DataFrame([features], columns=[
            "study_hours", "attendance", "sleep_hours", "internet_usage", 
            "assignments_completed", "previous_score", "exam_score"
        ])
        
        pred = int(model.predict(df_input)[0])
        prob = model.predict_proba(df_input)[0]
        
        status = "Placed" if pred == 1 else "Not Placed"
        confidence = round(prob[pred] * 100, 2)
        
    return render_template_string(HTML_TEMPLATE, status=status, confidence=confidence, model_error=model_error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


             
                       
