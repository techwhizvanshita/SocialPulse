# step2 :- python -m streamlit run "C:\Users\91860\Desktop\SocialPulse\app.py"
# step1 :- pip install --user pandas streamlit clean-text matplotlib wordcloud vaderSentiment langdetect googletrans==4.0.0-rc1 altair

import pandas as pd
import streamlit as st
import cleantext
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator
from functools import lru_cache

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
aspects = ['price', 'quality', 'service', 'delivery']

# -------------------------
# 🔹 Helper Functions
# -------------------------
@lru_cache(maxsize=10000)
def cached_translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return text
        else:
            return translator.translate(text, src=lang, dest='en').text
    except Exception:
        return text

def get_vader_sentiment_score(text):
    return analyzer.polarity_scores(str(text))['compound']

def analyze_vader_sentiment(score):
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def final_sentiment(row):
    rating = row['Rating']
    text_sent = row['text_sentiment']
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return text_sent
    elif rating <= 2:
        return text_sent
    else:
        return 'Neutral'

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# 🔹 Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🧠 Product Review Sentiment Analyzer")

# -------------------------
# 🔹 1. Single Text Analysis
# -------------------------
with st.expander('🔍 Single Text Analysis'):
    text = st.text_input('Enter a review:')
    if text:
        lang = detect(text)
        if lang != 'en':
            english_text = cached_translate_to_english(text)
            st.write('📝 English Translation:', english_text)
        else:
            english_text = text
        score = get_vader_sentiment_score(english_text)
        sentiment = analyze_vader_sentiment(score)
        st.write('📊 Sentiment:', sentiment)
        st.write('📊 VADER Score:', round(score, 2))
        aspect_mentions = {aspect: (aspect in english_text.lower()) for aspect in aspects}
        st.write('🔎 Aspect Mentions:', aspect_mentions)

    pre = st.text_input('Clean the text:')
    if pre:
        cleaned = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                  stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write('🧼 Cleaned Text:', cleaned)

# -------------------------
# 🔹 2. File Upload
# -------------------------
st.subheader("📂 Upload a Review CSV File")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# -------------------------
# 🔹 3. Main Analysis
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    if 'Review_Summary' not in df.columns or 'Rating' not in df.columns:
        st.error("❌ CSV must contain both 'Review_Summary' and 'Rating' columns.")
    else:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])

        # User option for translation
        st.sidebar.markdown("### Translation Options")
        translate_reviews = st.sidebar.checkbox("Translate non-English reviews to English", value=True)

        # Translation with progress bar and caching
        if translate_reviews:
            st.info("Translating reviews to English (if needed)...")
            english_reviews = []
            progress_bar = st.progress(0)
            review_list = df['Review_Summary'].astype(str).tolist()
            for i, review in enumerate(review_list):
                english_reviews.append(cached_translate_to_english(review))
                if len(review_list) > 1:
                    progress_bar.progress((i+1)/len(review_list))
            df['Review_Summary_English'] = english_reviews
            progress_bar.empty()
        else:
            df['Review_Summary_English'] = df['Review_Summary'].astype(str)

        # VADER sentiment analysis
        df['score'] = df['Review_Summary_English'].apply(get_vader_sentiment_score)
        df['text_sentiment'] = df['score'].apply(analyze_vader_sentiment)
        df['final_sentiment'] = df.apply(final_sentiment, axis=1)
        df['mismatch'] = df['text_sentiment'] != df['final_sentiment']

        # Aspect extraction
        for aspect in aspects:
            col_name = aspect + '_mention'
            df[col_name] = df['Review_Summary_English'].str.lower().str.contains(aspect)

        st.sidebar.title("📊 Filters")
        min_rating, max_rating = int(df['Rating'].min()), int(df['Rating'].max())
        selected_range = st.sidebar.slider("Select Rating Range", min_rating, max_rating, (min_rating, max_rating))
        df = df[(df['Rating'] >= selected_range[0]) & (df['Rating'] <= selected_range[1])]

        st.success(f"✅ Total reviews analyzed: {len(df)}")
        st.write(df.head())

        # -------------------------
        # 🔹 Sentiment Distribution
        # -------------------------
        st.subheader("📊 Sentiment Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df['final_sentiment'].value_counts())

        with col2:
            fig, ax = plt.subplots()
            ax.pie(df['final_sentiment'].value_counts(), labels=df['final_sentiment'].value_counts().index,
                   autopct="%1.1f%%", startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        # -------------------------
        # 🔹 Product-wise Breakdown
        # -------------------------
        if 'ProductName' in df.columns:
            st.subheader("🛍️ Product-wise Sentiment Breakdown")
            product_group = df.groupby(['ProductName', 'final_sentiment']).size().unstack(fill_value=0)
            st.bar_chart(product_group)

        # -------------------------
        # 🔹 Aspect Mentions Breakdown
        # -------------------------
        st.subheader("🔎 Aspect Mentions Breakdown")
        aspect_cols = [aspect + '_mention' for aspect in aspects]
        aspect_counts = df[aspect_cols].sum().rename(lambda x: x.replace('_mention', '').capitalize())
        st.write(aspect_counts)

        # -------------------------
        # 🔹 Word Cloud
        # -------------------------
        st.subheader("☁️ Word Cloud of All Reviews")
        all_text = " ".join(df['Review_Summary_English'].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # -------------------------
        # 🔹 Download Results
        # -------------------------
        st.subheader("⬇️ Download Sentiment CSV")
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

        # -------------------------
        # 🔹 Show Full Data
        # -------------------------
        with st.expander("🗂 Show All Reviews"):
            st.dataframe(df)
