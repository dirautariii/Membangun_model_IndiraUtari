print(">>> RUNNING FILE:", __file__)
import time
import psutil
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest
)

MODEL_PATH = r"C:\Users\INDRA\Membangun_model\mlruns\0\073bbb893e3742b2910137e64ca29b77\artifacts\model"
model_load_start = time.time()
model = mlflow.pyfunc.load_model(MODEL_PATH)
MODEL_LOAD_TIME = time.time() - model_load_start

app = Flask(__name__)



REQUEST_TOTAL = Counter(
    "ml_request_total",
    "Total request inference"
)

REQUEST_ERROR_TOTAL = Counter(
    "ml_request_error_total",
    "Total error inference"
)

PREDICTION_TOTAL = Counter(
    "ml_prediction_total",
    "Total prediksi berhasil"
)

REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Latency inference (detik)"
)

ACTIVE_REQUESTS = Gauge(
    "ml_active_requests",
    "Jumlah request aktif"
)

CPU_USAGE = Gauge(
    "ml_cpu_usage_percent",
    "Penggunaan CPU (%)"
)

MEMORY_USAGE = Gauge(
    "ml_memory_usage_percent",
    "Penggunaan RAM (%)"
)

MODEL_UP = Gauge(
    "ml_model_up",
    "Status model (1=up, 0=down)"
)

MODEL_LOAD_TIME_METRIC = Gauge(
    "ml_model_load_time_seconds",
    "Waktu load model (detik)"
)

INFERENCE_THROUGHPUT = Gauge(
    "ml_inference_throughput",
    "Inference per detik"
)

MODEL_UP.set(1)
MODEL_LOAD_TIME_METRIC.set(MODEL_LOAD_TIME)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.route("/invocations", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_TOTAL.inc()
    ACTIVE_REQUESTS.inc()

    try:
        payload = request.get_json()

        if "dataframe_split" in payload:
            df = pd.DataFrame(
                payload["dataframe_split"]["data"],
                columns=payload["dataframe_split"]["columns"]
            )
        elif "instances" in payload:
            df = pd.DataFrame(payload["instances"])
        else:
            REQUEST_ERROR_TOTAL.inc()
            return jsonify({"error": "Format input tidak valid"}), 400

        prediction = model.predict(df)

        PREDICTION_TOTAL.inc(len(prediction))

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        INFERENCE_THROUGHPUT.set(1 / latency if latency > 0 else 0)

        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        REQUEST_ERROR_TOTAL.inc()
        return jsonify({"error": str(e)}), 500

    finally:
        ACTIVE_REQUESTS.dec()

if __name__ == "__main__":
    print("=" * 60)
    print("MLFLOW MODEL SERVING + PROMETHEUS METRICS")
    print("Endpoint Inference : http://127.0.0.1:5002/invocations")
    print("Metrics Prometheus : http://127.0.0.1:5002/metrics")
    print("Health Check       : http://127.0.0.1:5002/ping")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5002)
