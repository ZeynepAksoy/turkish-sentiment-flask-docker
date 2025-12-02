from flask import Flask, request
from .model import predict_emotion

app = Flask(__name__)


HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Türkçe Duygu Analizi (Kaggle + LSTM)</title>
</head>
<body>
  <h1>Türkçe Duygu Analizi</h1>
  <p>Kaggle sad/happy veri setiyle eğitilmiş LSTM modeli</p>

  <form method="POST">
    <label for="text">Bir cümle yaz:</label><br>
    <input type="text" id="text" name="text" size="60" required value="{text}">
    <button type="submit">Tahmin Et</button>
  </form>
  {result}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result_html = ""

    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            label = predict_emotion(text)
            result_html = f"<h2>Model Tahmini: {label.upper()}</h2>"

    return HTML.format(text=text, result=result_html), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
