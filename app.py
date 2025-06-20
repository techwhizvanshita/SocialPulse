import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator
from functools import lru_cache
import re
import numpy as np

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
aspects = ['delivery', 'service', 'price', 'quality']
aspect_display = {
    'delivery': 'Delivery',
    'service': 'Customer Service',
    'price': 'Price',
    'quality': 'Product Quality'
}

# -------------------------
# Helper Functions
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

def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🧠 Product Review Sentiment Analyzer")

# -------------------------
# 1. Single Text Analysis
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
        aspect_mentions = {aspect_display.get(aspect, aspect.title()): (aspect in english_text.lower()) for aspect in aspects}
        st.write('🔎 Aspect Mentions:', aspect_mentions)

    pre = st.text_input('Clean the text:')
    if pre:
        cleaned = simple_clean(pre)
        st.write('🧼 Cleaned Text:', cleaned)

# -------------------------
# 2. File Upload
# -------------------------
st.subheader("📂 Upload a Review CSV File")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# -------------------------
# 3. Main Analysis
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
                sentiment_counts = df['final_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                color_scale = alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                        range=['#21ba45', '#a0a0a0', '#db2828'])  # green, gray, red

                bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x=alt.X('Sentiment', sort=['Positive', 'Neutral', 'Negative']),
                    y='Count',
                    color=alt.Color('Sentiment', scale=color_scale),
                    tooltip=['Sentiment', 'Count']
                ).properties(
                    width='container',
                    height=350
                )
                st.altair_chart(bar_chart, use_container_width=True)


        with col2:
                sentiment_counts = df['final_sentiment'].value_counts()
                # Ensure the order matches: Positive, Neutral, Negative
                order = ['Positive', 'Neutral', 'Negative']
                sentiment_counts = sentiment_counts.reindex(order).fillna(0)
                colors = ['green', 'grey', 'red']  # green for positive, grey for neutral, red for negative
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index,
                    autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis('equal')
                st.pyplot(fig)
      
        # -------------------------
        # Aspect Breakdown (Counts)
        # -------------------------
        aspect_breakdown = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            aspect_breakdown.append({
                'Aspect': aspect_display.get(aspect, aspect.title()),
                'Positive Count': pos,
                'Negative Count': neg,
                'Neutral Count': neu,
                'Total Mentions': total
            })
        aspect_breakdown_df = pd.DataFrame(aspect_breakdown)
        st.subheader("📊 Aspect Breakdown (Counts)")
        st.dataframe(aspect_breakdown_df)

        # -------------------------
        # Aspect-Based Sentiment Summary Table (Percentages)
        # -------------------------
        aspect_sentiment_summary = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            pos_pct = round(100 * pos / total, 1) if total else 0
            neg_pct = round(100 * neg / total, 1) if total else 0
            neu_pct = round(100 * neu / total, 1) if total else 0
            aspect_sentiment_summary.append({
                "Aspect": aspect_display.get(aspect, aspect.title()),
                "Positive (%)": pos_pct,
                "Negative (%)": neg_pct,
                "Neutral (%)": neu_pct
            })
        st.subheader("📋 Aspect-Based Sentiment Summary (%)")
        st.table(pd.DataFrame(aspect_sentiment_summary))

                # Find the aspect with the lowest positive and highest negative sentiment
        summary_df = pd.DataFrame(aspect_sentiment_summary)
        # Convert percentages to numbers for sorting
        summary_df['Positive (%)'] = pd.to_numeric(summary_df['Positive (%)'])
        summary_df['Negative (%)'] = pd.to_numeric(summary_df['Negative (%)'])

        # Identify the aspect needing most improvement
        weakest_aspect = summary_df.sort_values('Negative (%)', ascending=False).iloc[0]
        strongest_aspect = summary_df.sort_values('Positive (%)', ascending=False).iloc[0]

        # Simple AI-like recommendation logic
        improvement_threshold = 40  # you can adjust this
        if weakest_aspect['Negative (%)'] > improvement_threshold:
            suggestion = (f"⚠️ **Improvement Needed:** Customers are most dissatisfied with **{weakest_aspect['Aspect']}** "
                        f"({weakest_aspect['Negative (%)']}% negative sentiment). "
                        f"Consider addressing issues related to this aspect.")
        else:
            suggestion = (f"👍 **Overall Positive:** No single aspect stands out as highly negative. "
                        f"Continue monitoring for trends.")

        strength = (f"🌟 **Strength:** Customers are happiest with **{strongest_aspect['Aspect']}** "
                    f"({strongest_aspect['Positive (%)']}% positive sentiment). "
                    f"Leverage this in your marketing and maintain quality.")

        # Display AI summary
        st.markdown("## 🤖 AI Analysis & Recommendations")
        st.write(suggestion)
        st.write(strength)

        # # -------------------------
        # # Word Cloud (with stopwords and product names)
        # # -------------------------
        # st.subheader("☁️ Word Cloud of All Reviews")
        # stopwords = set(STOPWORDS)
        # if 'ProductName' in df.columns:
        #     prod_names = df['ProductName'].astype(str).str.lower().unique()
        #     stopwords.update(prod_names)
        # all_text = " ".join(df['Review_Summary_English'].astype(str).tolist())
        # wordcloud = WordCloud(
        #     width=800,
        #     height=400,
        #     background_color='white',
        #     stopwords=stopwords
        # ).generate(all_text)
        # fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        # ax_wc.imshow(wordcloud, interpolation='bilinear')
        # ax_wc.axis('off')
        # st.pyplot(fig_wc)

        # -------------------------
        # Download Results
        # -------------------------
        st.subheader("⬇️ Download Sentiment CSV")
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

        # -------------------------
        # Show Full Data
        # -------------------------
        with st.expander("🗂 Show All Reviews"):
            st.dataframe(df)
