import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="📚 AI Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .book-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .search-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model and data"""
    try:
        # Load model components
        model = pkl.load(open("model.pkl", "rb"))
        books_name = pkl.load(open("books_name.pkl", "rb"))
        final_ratings = pkl.load(open("final_ratings.pkl", "rb"))
        book_pivot = pkl.load(open("book_pivot.pkl", "rb"))
        
        return model, books_name, final_ratings, book_pivot, None
    except FileNotFoundError as e:
        return None, None, None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, None, None, f"Error loading model: {e}"

def get_book_recommendations(book_name, model, book_pivot, n_recommendations=5):
    """Get book recommendations"""
    try:
        # Find the book's row index in the pivot table
        book_indices = np.where(book_pivot.index == book_name)[0]
        if len(book_indices) == 0:
            return [], f"Book '{book_name}' not found in database."
        
        book_id = book_indices[0]
        
        # Create query vector
        query = book_pivot.iloc[book_id, :].fillna(0).values.reshape(1, -1)
        
        # Get nearest neighbors
        distances, indices = model.kneighbors(query, n_neighbors=n_recommendations + 1)
        
        # Collect recommendations (excluding the input book)
        recommendations = []
        for i in range(1, len(indices[0])):
            book_title = book_pivot.index[indices[0][i]]
            similarity_score = 1 - distances[0][i]  # Convert distance to similarity
            recommendations.append((book_title, similarity_score))
        
        return recommendations, None
    except Exception as e:
        return [], f"Error getting recommendations: {e}"

def get_book_details(book_title, final_ratings):
    """Get details about a specific book"""
    try:
        book_data = final_ratings[final_ratings['Title'] == book_title].iloc[0]
        return {
            'title': book_data['Title'],
            'author': book_data['Author'],
            'year': book_data['Year'],
            'publisher': book_data['Publisher'],
            'avg_rating': final_ratings[final_ratings['Title'] == book_title]['Book-Rating'].mean(),
            'num_ratings': book_data['num_of_rating'],
            'image_url': book_data.get('Image_URL', '')
        }
    except:
        return None

def create_rating_distribution_chart(final_ratings, selected_books):
    """Create a rating distribution chart for selected books"""
    if not selected_books:
        return None
    
    fig = go.Figure()
    
    for book in selected_books:
        book_ratings = final_ratings[final_ratings['Title'] == book]['Book-Rating']
        fig.add_trace(go.Histogram(
            x=book_ratings,
            name=book[:30] + "..." if len(book) > 30 else book,
            opacity=0.7,
            nbinsx=10
        ))
    
    fig.update_layout(
        title="Rating Distribution Comparison",
        xaxis_title="Rating",
        yaxis_title="Number of Users",
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_popularity_chart(final_ratings):
    """Create a chart showing most popular books"""
    popular_books = (final_ratings.groupby('Title')
                    .agg({
                        'Book-Rating': 'mean',
                        'num_of_rating': 'first'
                    })
                    .reset_index())
    
    popular_books = popular_books.sort_values('num_of_rating', ascending=False).head(20)
    
    fig = px.scatter(
        popular_books,
        x='num_of_rating',
        y='Book-Rating',
        hover_data=['Title'],
        title="Book Popularity vs Average Rating",
        labels={'num_of_rating': 'Number of Ratings', 'Book-Rating': 'Average Rating'},
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI Book Recommender System</h1>', unsafe_allow_html=True)
    st.markdown("### Discover your next favorite book with AI-powered recommendations!")
    
    # Load model and data
    with st.spinner("🔄 Loading recommendation engine..."):
        model, books_name, final_ratings, book_pivot, error = load_model_and_data()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please make sure you have run the training script first to generate the model files.")
        st.code("""
# Run this first to train the model:
python train_model.py
        """)
        return
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("### 🔍 Model Information")
        st.info(f"📊 **Books in database:** {len(books_name):,}")
        st.info(f"👥 **Users analyzed:** {book_pivot.shape[1]:,}")
        st.info(f"⭐ **Total ratings:** {len(final_ratings):,}")
        
        # Display some statistics
        avg_rating = final_ratings['Book-Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}/10")
        
        most_rated_book = final_ratings.loc[final_ratings['num_of_rating'].idxmax(), 'Title']
        max_ratings = final_ratings['num_of_rating'].max()
        st.metric("Most Rated Book", f"{max_ratings} ratings")
        st.caption(f"📖 {most_rated_book[:50]}...")
        
        # Model algorithm info
        st.markdown("### 🤖 Algorithm")
        st.write("**K-Nearest Neighbors**")
        st.write("- Metric: Cosine Similarity")
        st.write("- Collaborative Filtering")
        st.write("- Memory-based approach")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.markdown("### 🔎 Find Book Recommendations")
        
        # Search methods
        search_method = st.radio(
            "Choose search method:",
            ["📝 Type book name", "📋 Select from list", "🎲 Random book"],
            horizontal=True
        )
        
        selected_book = None
        
        if search_method == "📝 Type book name":
            # Text input with autocomplete
            book_input = st.text_input(
                "Enter book title:",
                placeholder="e.g., Harry Potter, The Alchemist, 1984...",
                help="Start typing and we'll help you find the exact title"
            )
            
            if book_input:
                # Find matching books
                matching_books = [book for book in books_name if book_input.lower() in book.lower()]
                
                if matching_books:
                    if len(matching_books) == 1:
                        selected_book = matching_books[0]
                        st.success(f"✅ Found: {selected_book}")
                    else:
                        st.info(f"Found {len(matching_books)} matching books:")
                        selected_book = st.selectbox("Select the correct book:", matching_books)
                else:
                    st.warning("❌ No matching books found. Try a different search term.")
        
        elif search_method == "📋 Select from list":
            # Dropdown selection
            popular_books = (final_ratings.groupby('Title')['num_of_rating']
                            .first().sort_values(ascending=False).head(100).index.tolist())
            
            selected_book = st.selectbox(
                "Choose from popular books:",
                options=[""] + popular_books,
                format_func=lambda x: "Select a book..." if x == "" else x
            )
            
            if selected_book:
                st.success(f"✅ Selected: {selected_book}")
        
        elif search_method == "🎲 Random book":
            if st.button("🎲 Get Random Book"):
                selected_book = np.random.choice(books_name)
                st.success(f"🎯 Random pick: {selected_book}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get recommendations
        if selected_book and selected_book != "":
            st.markdown("---")
            
            # Display selected book details
            book_details = get_book_details(selected_book, final_ratings)
            if book_details:
                st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
                st.markdown(f"### 📖 Selected Book")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**Title:** {book_details['title']}")
                    st.write(f"**Author:** {book_details['author']}")
                    st.write(f"**Publisher:** {book_details['publisher']}")
                    st.write(f"**Year:** {book_details['year']}")
                with col_b:
                    st.metric("Avg Rating", f"{book_details['avg_rating']:.1f}/10")
                    st.metric("Total Ratings", f"{book_details['num_ratings']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Number of recommendations slider
            num_recommendations = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=10,
                value=5,
                help="Choose how many book recommendations you want"
            )
            
            # Generate recommendations button
            if st.button("🚀 Get Recommendations", use_container_width=True):
                with st.spinner("🔮 Generating personalized recommendations..."):
                    recommendations, error = get_book_recommendations(
                        selected_book, model, book_pivot, num_recommendations
                    )
                
                if error:
                    st.error(f"❌ {error}")
                elif recommendations:
                    st.markdown("### 🎯 Your Personalized Recommendations")
                    
                    # Display recommendations
                    for i, (book_title, similarity) in enumerate(recommendations, 1):
                        rec_details = get_book_details(book_title, final_ratings)
                        
                        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                        
                        col_rec1, col_rec2 = st.columns([4, 1])
                        with col_rec1:
                            st.markdown(f"### {i}. {book_title}")
                            if rec_details:
                                st.write(f"**Author:** {rec_details['author']}")
                                st.write(f"**Year:** {rec_details['year']}")
                                st.write(f"**Publisher:** {rec_details['publisher']}")
                        
                        with col_rec2:
                            st.metric("Match", f"{similarity*100:.1f}%")
                            if rec_details:
                                st.metric("Rating", f"{rec_details['avg_rating']:.1f}/10")
                                st.metric("Reviews", f"{rec_details['num_ratings']}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Save recommendations to session state for analysis
                    st.session_state.last_recommendations = [book for book, _ in recommendations]
                    st.session_state.selected_book = selected_book
                else:
                    st.warning("⚠️ No recommendations found for this book.")
    
    with col2:
        st.markdown("### 📊 Analytics Dashboard")
        
        # Popular books chart
        st.plotly_chart(create_popularity_chart(final_ratings), use_container_width=True)
        
        # Rating distribution for last recommendations
        if hasattr(st.session_state, 'last_recommendations'):
            books_to_analyze = [st.session_state.selected_book] + st.session_state.last_recommendations[:3]
            rating_chart = create_rating_distribution_chart(final_ratings, books_to_analyze)
            if rating_chart:
                st.plotly_chart(rating_chart, use_container_width=True)
        
        # Top authors
        st.markdown("### 👨‍💼 Top Authors by Ratings")
        top_authors = (final_ratings.groupby('Author')['num_of_rating']
                      .sum().sort_values(ascending=False).head(10))
        
        for author, ratings in top_authors.items():
            st.write(f"**{author}** - {ratings:,} total ratings")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>📚 <b>AI Book Recommender</b> | Powered by Collaborative Filtering & K-Nearest Neighbors</p>
        <p>🔬 Built with Streamlit, scikit-learn, and Plotly | 💡 Discover books based on user preferences</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()