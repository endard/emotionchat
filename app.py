from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import tensorflow as tf
import pickle
import os
import random

# Load model dan vectorizer
model = tf.keras.models.load_model("model_emosi_tf.keras")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_cols = ["neutral", "negatif_intens", "negatif_sosial",
              "kognitif_keingintahuan", "kognitif_refleksi",
              "positif_gembira", "positif_afirmasi", "positif_kelekatan"]

RESPON_EMOSI = {
    "neutral": [
        "Semoga harimu baik-baik saja 😊",
        "Tetap semangat ya meski hari biasa",
        "Santai aja, kadang hening itu damai"
    ],
    "negatif_intens": [
        "Kamu tampaknya sedang merasa sangat tertekan. Ingin bercerita lebih banyak?",
        "Napas dulu... kamu nggak sendiri.",
        "Kalau kamu butuh didengar, aku di sini kok."
    ],
    "negatif_sosial": [
        "Kelihatannya kamu sedang tidak nyaman secara sosial. Kami di sini untuk mendengarkan.",
        "Ada yang bikin kamu nggak enak dengan orang lain?",
        "Lingkungan kadang melelahkan ya... 😔"
    ],
    "kognitif_keingintahuan": [
        "Rasa penasaranmu bagus! Ayo eksplor lebih banyak.",
        "Pertanyaanmu menarik! Yuk bahas lebih dalam.",
        "Suka banget liat kamu berpikir kritis 😄"
    ],
    "kognitif_refleksi": [
        "Kamu terlihat sedang merenung atau berpikir dalam. Ada yang bisa kami bantu?",
        "Kadang berpikir sendiri bikin kita sadar banyak hal ya.",
        "Kamu lagi refleksi ya? Bagus banget itu."
    ],
    "positif_gembira": [
        "Wah! Senang mendengar itu! 😄",
        "Kebahagiaanmu menular loh!",
        "Yay! Good vibes detected! 🎉"
    ],
    "positif_afirmasi": [
        "Terus semangat ya! Kamu hebat.",
        "Kamu punya potensi besar loh!",
        "Yakin deh kamu bisa 💪"
    ],
    "positif_kelekatan": [
        "Terima kasih atas perhatianmu. Kamu tidak sendirian.",
        "Senang banget kamu berbagi hal ini.",
        "Kita saling dukung ya 🤝"
    ]
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
    respon_list = RESPON_EMOSI.get(emosi, RESPON_EMOSI["neutral"])
    respon = random.choice(respon_list)
    await update.message.reply_text(respon)

def main():
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
