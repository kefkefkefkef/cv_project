import streamlit as st
from PIL import Image
def main():
    results1 = Image.open('yolometrics/results1.png')
    results2 = Image.open('yolometrics/results2.png')
    results3 = Image.open('yolometrics/results3.png')
    results4 = Image.open('yolometrics/results4.png')

    prcurve1 = Image.open('yolometrics/PR_curve1.png')
    prcurve2 = Image.open('yolometrics/PR_curve2.png')
    prcurve3 = Image.open('yolometrics/PR_curve3.png')
    prcurve4 = Image.open('yolometrics/PR_curve4.png')

    st.image(results1, caption='25 epochs')
    st.image(results2, caption='50 epochs')
    st.image(results3, caption='100 epochs')
    st.image(results4, caption='200 epochs')
    st.image(prcurve1, caption='25 epochs')
    st.image(prcurve2, caption='50 epochs')
    st.image(prcurve3, caption='100 epochs')
    st.image(prcurve4, caption='200 epochs')
if __name__ == '__main__':
    main()
