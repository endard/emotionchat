from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import tensorflow as tf
import pickle
import os

# Load model dan vectorizer
model = tf.keras.models.load_model("model_emosi_tf.keras")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_cols = ["neutral", "negatif_intens", "negatif_sosial",
              "kognitif_keingintahuan", "kognitif_refleksi",
              "positif_gembira", "positif_afirmasi", "positif_kelekatan"]

RESPON_EMOSI = {
    "neutral": "Semoga harimu baik-baik saja 😊",
    "negatif_intens": "Kamu tampaknya sedang merasa sangat tertekan. Ingin bercerita lebih banyak?",
    "negatif_sosial": "Kelihatannya kamu sedang tidak nyaman secara sosial. Kami di sini untuk mendengarkan.",
    "kognitif_keingintahuan": "Rasa penasaranmu bagus! Ayo eksplor lebih banyak.",
    "kognitif_refleksi": "Kamu terlihat sedang merenung atau berpikir dalam. Ada yang bisa kami bantu?",
    "positif_gembira": "Wah! Senang mendengar itu! 😄",
    "positif_afirmasi": "Terus semangat ya! Kamu hebat.",
    "positif_kelekatan": "Terima kasih atas perhatianmu. Kamu tidak sendirian."
}

def prediksi_emosi(teks):
    vec = vectorizer(tf.constant([teks]))
    pred = model.predict(vec)[0]
    threshold = 0.3
    hasil = [(label_cols[i], float(pred[i])) for i in range(len(pred)) if pred[i] >= threshold]
    if hasil:
        hasil.sort(key=lambda x: x[1], reverse=True)
        return hasil[0][0]
    return "neutral"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    teks = update.message.text
    emosi = prediksi_emosi(teks)
    respon = RESPON_EMOSI.get(emosi, RESPON_EMOSI["neutral"])
    await update.message.reply_text(respon)

def main():
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
