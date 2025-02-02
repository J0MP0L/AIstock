import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("NVDA Stock Price Prediction")

import yfinance as yf

# กำหนดสัญลักษณ์หุ้น (ตัวอย่าง: AAPL - Apple)
symbol = "NVDA"

# ดึงข้อมูลย้อนหลัง 10 วัน
data = yf.download(symbol, period="11d", interval="1d")

df=pd.DataFrame()
df["Open"]=np.array(data["Open"]["NVDA"])
df["High"]=np.array(data["High"]["NVDA"])
df["Low"]=np.array(data["Low"]["NVDA"])
df["Close"]=np.array(data["Close"]["NVDA"])
df["Volume"]=np.array(data["Volume"]["NVDA"])

df["diff_n_o"]=[None]+[np.log(df["Open"][i+1])-np.log(df["Open"][i]) for i in (range(len(df)-1))]
df["diff_n_h"]=[None]+[np.log(df["High"][i+1])-np.log(df["High"][i]) for i in (range(len(df)-1))]
df["diff_n_c"]=[None]+[np.log(df["Close"][i+1])-np.log(df["Close"][i]) for i in (range(len(df)-1))]
df["diff_n_v"]=[None]+[np.log(df["Volume"][i+1])-np.log(df["Volume"][i]) for i in (range(len(df)-1))]
df["diff_n_l"]=[None]+[np.log(df["Low"][i+1])-np.log(df["Low"][i]) for i in (range(len(df)-1))]
df["diff_n_CO"]=[None]+[np.log(df["Close"][i+1])-np.log(df["Open"][i+1]) for i in (range(len(df)-1))]
data2=df.dropna()
Xl1=data2[["diff_n_o","diff_n_l","diff_n_h","diff_n_c","diff_n_v","diff_n_CO"]]


from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom", name="acc")
def acc(y_true, y_pred):
    YT = tf.cast(y_true > 0, tf.int8)
    YP = tf.cast(y_pred > 0, tf.int8)
    accuracy = tf.reduce_mean(tf.cast(YT == YP, tf.float32))
    return accuracy
# Load pre-trained RNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('AI.keras', custom_objects={
    'acc': acc,
})

model = load_model()

@st.cache_resource
def load_model3():
    return tf.keras.models.load_model('AI3.keras', custom_objects={
    'acc': acc,
})

model3 = load_model3()

@st.cache_resource
def load_model4():
    return tf.keras.models.load_model('AI4.keras', custom_objects={
    'acc': acc,
})

model4 = load_model4()

if st.button("Run model1"):
    st.write("### Past 10 Days' Data:")
    st.write(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    for stock in ["Open", "High", "Low", "Close"]:
        ax.plot(data.index, df[stock], label=stock)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Comparison")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    predicted_price = model.predict(Xl1.values[np.newaxis])
    st.write(f"### Predicted Next Day High Price: {np.exp(predicted_price[0][0]+np.log(df["High"].iloc[-1]))}")
    if predicted_price[0][0] > 0:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มเพิ่มขึ้น เนื่องจากผลการทำนายราคาสูงสุดมีค่าสูงกว่าเมื่อวาน")
    else:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มลดลง เนื่องจากผลการทำนายราคาสูงสุดมีค่าต่ำกว่าเมื่อวาน")

if st.button("Run model2"):
    st.write("### Past 10 Days' Data:")
    st.write(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    for stock in ["Open", "High", "Low", "Close"]:
        ax.plot(data.index, df[stock], label=stock)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Comparison")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    predicted_price = model3.predict(Xl1.values[np.newaxis])
    st.write(f"### Predicted Next Day High Price: {np.exp(predicted_price[0][0]+np.log(df["High"].iloc[-1]))}")
    if predicted_price[0][0] > 0:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มเพิ่มขึ้น เนื่องจากผลการทำนายราคาสูงสุดมีค่าสูงกว่าเมื่อวาน")
    else:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มลดลง เนื่องจากผลการทำนายราคาสูงสุดมีค่าต่ำกว่าเมื่อวาน")
    
if st.button("Run model2"):
    st.write("### Past 10 Days' Data:")
    st.write(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    for stock in ["Open", "High", "Low", "Close"]:
        ax.plot(data.index, df[stock], label=stock)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Comparison")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    predicted_price = model4.predict(Xl1.values[np.newaxis])
    st.write(f"### Predicted Next Day High Price: {np.exp(predicted_price[0][0]+np.log(df["High"].iloc[-1]))}")
    if predicted_price[0][0] > 0:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มเพิ่มขึ้น เนื่องจากผลการทำนายราคาสูงสุดมีค่าสูงกว่าเมื่อวาน")
    else:
        st.write("# ราคาหุ้นในวันนี้มีแนวโน้มลดลง เนื่องจากผลการทำนายราคาสูงสุดมีค่าต่ำกว่าเมื่อวาน")
