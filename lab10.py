import streamlit as st
from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError
from funcs import *

st.set_option('deprecation.showfileUploaderEncoding', False)
db = pd.read_csv('out.csv', delimiter=',')
db['vector'] = db['vector'].apply(
    lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.float64))


def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = ImageOps.exif_transpose(image)
        st.image(image)
        img_opencv = np.array(image)
        img_opencv = img_opencv[:, :, ::-1].copy()
        links = get_neighbours_links(db, get_k_neighbours(vectorize(img_opencv), db, 3))
        st.success("There are similar images:")
        col = st.columns(3)
        for i in range(len(links)):
            try:
                with col[i]:
                    similar_image = Image.open('images/' + links[i])
                    st.image(similar_image, width=200)
            except UnidentifiedImageError:
                pass


if __name__ == '__main__':
    main()
