import torch
from model import Model
from predictor import Predictor
import streamlit as st 
import matplotlib.pyplot as plt
from PIL import Image 

model = Model()
model.load_state_dict(torch.load("model_cnn.pth", map_location = torch.device("cpu"))) #훈련한 파라미터를 읽어들임
predictor = Predictor(model)


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("이미지 인식 앱")
st.sidebar.write("이미지 인식 모델을 사용해서 무슨 이미지인지를 판정 합니다.")

st.sidebar.write("")

img_source = st.sidebar.radio("이미지 소스를 선택해 주세요.", ("이미지를 업로드", "카메라로 촬영"))

if img_source == "이미지를 업로드":
    img_file = st.sidebar.file_uploader("이미지를 선택해 주세요.", type = ["png", "jpg", "jpeg"])
elif img_source == "카메라로 촬영":
    img_file = st.camera_input("카메라로 촬영")

if(img_file is not None):
    with st.spinner("측정 중..."):
        img = Image.open(img_file)
        st.image(img, caption = "대상 이미지", width = 480)
        st.write("")
        
        results = predictor.predict(img)
        st.subheader("판정 결과")
        n_top = 3 #확률이 높은 3개만
        
        for result in results[:n_top]:
            st.write(f"{round(result[2] * 100, 2)}%의 확률로 {result[0]}입니다.")
            
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
    
        fig, ax = plt.subplots()
        wedgeprops = {"width" : 0.3, "edgecolor" : "white"}
        textprops = {"fontsize" : 6}
        ax.pie(pie_probs, labels = pie_labels, counterclock= False, startangle = 90, textprops = textprops, autopct = "%.2f", wedgeprops=wedgeprops)
        
        st.pyplot(fig) #그래프 그림
        st.sidebar.write("")
        st.sidebar.write("")
        
        st.sidebar.caption("""이 앱은 Fashion-MNIST를 훈련 데이터로 사용하고 있습니다.\n
                           Copyright (c) 2017 Zalando SE\n
                           Released under the MIT license\n
                           https://github.com/zalandoresearch/fashion-mnist#/license""")