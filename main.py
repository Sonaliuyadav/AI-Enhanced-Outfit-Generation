# # from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
# # from werkzeug.security import generate_password_hash, check_password_hash
# # import pandas as pd
# # import numpy as np
# # import ast
# # import os
# # from diffusers import StableDiffusionPipeline
# # import torch
# # from sklearn.metrics.pairwise import cosine_similarity
# # import sqlite3
# # from datetime import datetime
# # import secrets
# # from PIL import Image
# # import base64
# # from io import BytesIO
# # import logging

# # app = Flask(__name__)
# # app.secret_key = secrets.token_hex(16)

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Global variables for the ML model
# # fashion_pipeline = None
# # fashion_df = pd.DataFrame()

# # def load_model():
# #     """Initialize the ML model and load fashion dataset"""
# #     global fashion_pipeline, fashion_df
# #     try:
# #         # Use a more efficient model loading approach
# #         fashion_pipeline = StableDiffusionPipeline.from_pretrained(
# #             "runwayml/stable-diffusion-v1-5", 
# #             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
# #             use_safetensors=True
# #         )
        
# #         # Enable memory efficient attention
# #         if torch.cuda.is_available():
# #             fashion_pipeline = fashion_pipeline.to("cuda")
# #             fashion_pipeline.enable_attention_slicing()
# #             fashion_pipeline.enable_xformers_memory_efficient_attention()
# #         else:
# #             fashion_pipeline = fashion_pipeline.to("cpu")
            
# #         logger.info("Fashion pipeline loaded successfully")
# #     except Exception as e:
# #         logger.error(f"Error loading fashion pipeline: {e}")
# #         fashion_pipeline = None
    
# #     # Load fashion dataset
# #     try:
# #         fashion_df = pd.read_csv("updated_recommendation.csv")
# #         fashion_df['product_attributes'] = fashion_df['product_attributes'].apply(
# #             lambda x: ast.literal_eval(x) if pd.notna(x) else {}
# #         )
# #         logger.info(f"Fashion dataset loaded with {len(fashion_df)} items")
# #     except Exception as e:
# #         logger.error(f"Error loading fashion dataset: {e}")
# #         fashion_df = pd.DataFrame()

# # def init_db():
# #     conn = sqlite3.connect('fashion_app.db')
# #     cursor = conn.cursor()
    
# #     cursor.execute('''
# #         CREATE TABLE IF NOT EXISTS users (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             username TEXT UNIQUE NOT NULL,
# #             email TEXT UNIQUE NOT NULL,
# #             password_hash TEXT NOT NULL,
# #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# #         )
# #     ''')
    
# #     cursor.execute('''
# #         CREATE TABLE IF NOT EXISTS search_history (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             user_id INTEGER,
# #             search_prompt TEXT NOT NULL,
# #             search_type TEXT NOT NULL DEFAULT 'recommendation',
# #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# #             FOREIGN KEY (user_id) REFERENCES users (id)
# #         )
# #     ''')
    
# #     cursor.execute('''
# #         CREATE TABLE IF NOT EXISTS favorites (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             user_id INTEGER,
# #             product_name TEXT NOT NULL,
# #             product_details TEXT,
# #             image_data TEXT,
# #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# #             FOREIGN KEY (user_id) REFERENCES users (id)
# #         )
# #     ''')
    
# #     cursor.execute('''
# #         CREATE TABLE IF NOT EXISTS generated_images (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             user_id INTEGER,
# #             prompt TEXT NOT NULL,
# #             image_data TEXT NOT NULL,
# #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# #             FOREIGN KEY (user_id) REFERENCES users (id)
# #         )
# #     ''')
    
# #     conn.commit()
# #     conn.close()

# # def get_db_connection():
# #     conn = sqlite3.connect('fashion_app.db')
# #     conn.row_factory = sqlite3.Row
# #     return conn

# # def get_fashion_recommendations(category, subcategory=None, gender=None, usage=None, color=None, limit=12):
# #     """Get fashion recommendations based on filters using cosine similarity"""
# #     global fashion_df
    
# #     if fashion_df.empty:
# #         return []
    
# #     try:
# #         # Start with the full dataset
# #         filtered_df = fashion_df.copy()
        
# #         # Apply filters
# #         if category and category != 'All':
# #             filtered_df = filtered_df[filtered_df['masterCategory'].str.contains(category, case=False, na=False)]
        
# #         if subcategory and subcategory != 'All':
# #             filtered_df = filtered_df[filtered_df['subCategory'].str.contains(subcategory, case=False, na=False)]
            
# #         if gender and gender != 'All':
# #             filtered_df = filtered_df[
# #                 (filtered_df['gender'].str.contains(gender, case=False, na=False)) |
# #                 (filtered_df['gender'].str.contains('Unisex', case=False, na=False))
# #             ]
            
# #         if usage and usage != 'All':
# #             filtered_df = filtered_df[filtered_df['usage'].str.contains(usage, case=False, na=False)]
            
# #         if color and color != 'All':
# #             filtered_df = filtered_df[filtered_df['baseColour'].str.contains(color, case=False, na=False)]
        
# #         if filtered_df.empty:
# #             return []
        
    
# #         feature_cols = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
# #         available_cols = [col for col in feature_cols if col in filtered_df.columns]
        
# #         if available_cols:
# #             # Fill NaN values and convert to string
# #             for col in available_cols:
# #                 filtered_df[col] = filtered_df[col].fillna('Unknown').astype(str)
            
# #             # Create dummy variables for similarity calculation
# #             features = pd.get_dummies(filtered_df[available_cols])
            
# #             if len(features) > 1:
# #                 # Calculate cosine similarity matrix
# #                 similarity_matrix = cosine_similarity(features)
# #                 # Calculate average similarity score for each item
# #                 avg_similarity = similarity_matrix.mean(axis=1)
# #                 filtered_df['similarity_score'] = avg_similarity
# #             else:
# #                 filtered_df['similarity_score'] = 1.0
# #         else:
# #             filtered_df['similarity_score'] = 1.0
        
# #         # Sort by similarity score and get top results
# #         top_products = filtered_df.nlargest(limit, 'similarity_score')
        
# #         recommendations = []
# #         for _, product in top_products.iterrows():
# #             recommendations.append({
# #                 'name': product.get('productDisplayName', 'Fashion Item'),
# #                 'category': product.get('masterCategory', 'Fashion'),
# #                 'subcategory': product.get('subCategory', ''),
# #                 'article_type': product.get('articleType', ''),
# #                 'color': product.get('baseColour', 'Multi'),
# #                 'usage': product.get('usage', 'Casual'),
# #                 'season': product.get('season', 'All'),
# #                 'image': f"images/{product.get('image', 'placeholder.jpg')}",
# #                 'similarity_score': float(product.get('similarity_score', 0))
# #             })
        
# #         return recommendations
        
# #     except Exception as e:
# #         logger.error(f"Error in get_fashion_recommendations: {e}")
# #         return []

# # def generate_fashion_image(prompt, num_inference_steps=15):
# #     """Generate fashion image using Stable Diffusion with optimized settings"""
# #     global fashion_pipeline
    
# #     if fashion_pipeline is None:
# #         return None, "AI model not loaded. Please try again later."
    
# #     try:
# #         # Clean and enhance the prompt for better fashion results
# #         clean_prompt = prompt.strip()
        
# #         # Add fashion-specific keywords for better results
# #         enhanced_prompt = f"high quality fashion photography, {clean_prompt}, professional styling, clean background, detailed, 4k resolution"
        
# #         # Limit prompt length to avoid issues
# #         if len(enhanced_prompt) > 250:
# #             enhanced_prompt = f"fashion photography, {clean_prompt}, professional, detailed"
        
# #         logger.info(f"Generating image with enhanced prompt: {enhanced_prompt[:100]}...")
        
# #         # Generate image with optimized settings for speed and quality
# #         with torch.no_grad():
# #             # Clear GPU cache if using CUDA
# #             if torch.cuda.is_available():
# #                 torch.cuda.empty_cache()
            
# #             # Generate with reduced steps for faster generation
# #             result = fashion_pipeline(
# #                 enhanced_prompt,
# #                 num_inference_steps=num_inference_steps,  # Reduced from 20 to 15
# #                 guidance_scale=7.0,  # Slightly reduced for faster generation
# #                 height=512,
# #                 width=512,
# #                 generator=torch.Generator().manual_seed(42)  # Fixed seed for consistency
# #             )
            
# #             if not result.images or len(result.images) == 0:
# #                 return None, "No image generated. Please try a different prompt."
            
# #             image = result.images[0]
        
# #         # Convert to base64 for web display
# #         buffered = BytesIO()
# #         image.save(buffered, format="PNG", optimize=True, quality=85)
# #         img_str = base64.b64encode(buffered.getvalue()).decode()
        
# #         logger.info("Image generation completed successfully")
# #         return img_str, None
        
# #     except torch.cuda.OutOfMemoryError:
# #         logger.error("GPU out of memory")
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()
# #         return None, "GPU memory full. Please try a shorter prompt or try again later."
    
# #     except Exception as e:
# #         logger.error(f"Error in generate_fashion_image: {e}")
# #         error_msg = str(e)
        
# #         # Provide user-friendly error messages
# #         if "CUDA" in error_msg:
# #             return None, "GPU error. Please try again."
# #         elif "memory" in error_msg.lower():
# #             return None, "Memory error. Please try a shorter prompt."
# #         elif "timeout" in error_msg.lower():
# #             return None, "Generation timed out. Please try a simpler prompt."
# #         else:
# #             return None, "Generation failed. Please try a different prompt or try again later."

# # @app.route('/')
# # def home():
# #     return render_template('home.html')

# # @app.route('/signup', methods=['GET', 'POST'])
# # def signup():
# #     if request.method == 'POST':
# #         data = request.get_json()
# #         username = data.get('username')
# #         email = data.get('email')
# #         password = data.get('password')
        
# #         if not username or not email or not password:
# #             return jsonify({'success': False, 'message': 'All fields are required'})
        
# #         conn = get_db_connection()
        
# #         # Check if user already exists
# #         existing_user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
# #                                    (username, email)).fetchone()
        
# #         if existing_user:
# #             conn.close()
# #             return jsonify({'success': False, 'message': 'User already exists'})
        
# #         # Create new user
# #         password_hash = generate_password_hash(password)
# #         conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
# #                     (username, email, password_hash))
# #         conn.commit()
# #         conn.close()
        
# #         return jsonify({'success': True, 'message': 'Account created successfully'})
    
# #     return render_template('auth.html', mode='signup')

# # @app.route('/login', methods=['GET', 'POST'])
# # def login():
# #     if request.method == 'POST':
# #         data = request.get_json()
# #         username = data.get('username')
# #         password = data.get('password')
        
# #         if not username or not password:
# #             return jsonify({'success': False, 'message': 'Username and password required'})
        
# #         conn = get_db_connection()
# #         user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
# #                           (username, username)).fetchone()
# #         conn.close()
        
# #         if user and check_password_hash(user['password_hash'], password):
# #             session['user_id'] = user['id']
# #             session['username'] = user['username']
# #             return jsonify({'success': True, 'message': 'Login successful'})
# #         else:
# #             return jsonify({'success': False, 'message': 'Invalid credentials'})
    
# #     return render_template('auth.html', mode='login')

# # @app.route('/logout')
# # def logout():
# #     session.clear()
# #     return redirect(url_for('home'))

# # @app.route('/dashboard')
# # def dashboard():
# #     if 'user_id' not in session:
# #         return redirect(url_for('login'))
    
# #     # Initialize the model if not already loaded
# #     global fashion_pipeline, fashion_df
# #     if fashion_pipeline is None and fashion_df.empty:
# #         load_model()
    
# #     conn = get_db_connection()
# #     recent_searches = conn.execute(
# #         'SELECT * FROM search_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
# #         (session['user_id'],)
# #     ).fetchall()
    
# #     favorites = conn.execute(
# #         'SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
# #         (session['user_id'],)
# #     ).fetchall()
    
# #     generated_images = conn.execute(
# #         'SELECT * FROM generated_images WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
# #         (session['user_id'],)
# #     ).fetchall()
# #     conn.close()
    
# #     return render_template('dashboard.html', 
# #                          recent_searches=recent_searches, 
# #                          favorites=favorites,
# #                          generated_images=generated_images)

# # @app.route('/recommend')
# # def recommend():
# #     if 'user_id' not in session:
# #         return redirect(url_for('login'))
    
# #     # Get filter options from dataset
# #     global fashion_df
# #     if fashion_df.empty:
# #         load_model()
    
# #     categories = ['All'] + sorted(fashion_df['masterCategory'].dropna().unique().tolist())
# #     genders = ['All'] + sorted(fashion_df['gender'].dropna().unique().tolist())
# #     colors = ['All'] + sorted(fashion_df['baseColour'].dropna().unique().tolist())
# #     usages = ['All'] + sorted(fashion_df['usage'].dropna().unique().tolist())
    
# #     return render_template('recommend.html', 
# #                          categories=categories,
# #                          genders=genders, 
# #                          colors=colors,
# #                          usages=usages)

# # @app.route('/generate')
# # def generate():
# #     if 'user_id' not in session:
# #         return redirect(url_for('login'))
# #     return render_template('generate.html')

# # @app.route('/api/get_subcategories')
# # def get_subcategories():
# #     category = request.args.get('category')
# #     global fashion_df
    
# #     if category and category != 'All' and not fashion_df.empty:
# #         subcategories = fashion_df[
# #             fashion_df['masterCategory'].str.contains(category, case=False, na=False)
# #         ]['subCategory'].dropna().unique().tolist()
# #         subcategories = ['All'] + sorted(subcategories)
# #     else:
# #         subcategories = ['All']
    
# #     return jsonify(subcategories)

# # @app.route('/api/search_recommendations', methods=['POST'])
# # def search_recommendations():
# #     if 'user_id' not in session:
# #         return jsonify({'success': False, 'message': 'Please login first'})
    
# #     data = request.get_json()
# #     category = data.get('category', 'All')
# #     subcategory = data.get('subcategory', 'All')
# #     gender = data.get('gender', 'All')
# #     usage = data.get('usage', 'All')
# #     color = data.get('color', 'All')
    
# #     # Save search to history
# #     search_prompt = f"Category: {category}, Gender: {gender}, Usage: {usage}, Color: {color}"
# #     conn = get_db_connection()
# #     conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
# #                 (session['user_id'], search_prompt, 'recommendation'))
# #     conn.commit()
# #     conn.close()
    
# #     # Get recommendations
# #     recommendations = get_fashion_recommendations(category, subcategory, gender, usage, color)
    
# #     return jsonify({
# #         'success': True, 
# #         'recommendations': recommendations
# #     })

# # @app.route('/api/generate_image', methods=['POST'])
# # def generate_image():
# #     if 'user_id' not in session:
# #         return jsonify({'success': False, 'message': 'Please login first'})
    
# #     data = request.get_json()
# #     prompt = data.get('prompt', '').strip()
    
# #     if not prompt:
# #         return jsonify({'success': False, 'message': 'Please enter a prompt'})
    
# #     # Generate image
# #     image_data, error = generate_fashion_image(prompt)
    
# #     if error:
# #         return jsonify({'success': False, 'message': f'Error generating image: {error}'})
    
# #     # Save to database
# #     conn = get_db_connection()
# #     conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
# #                 (session['user_id'], prompt, 'generation'))
    
# #     conn.execute('INSERT INTO generated_images (user_id, prompt, image_data) VALUES (?, ?, ?)',
# #                 (session['user_id'], prompt, image_data))
# #     conn.commit()
# #     conn.close()
    
# #     return jsonify({
# #         'success': True, 
# #         'image': image_data,
# #         'prompt': prompt
# #     })

# # @app.route('/api/add_favorite', methods=['POST'])
# # def add_favorite():
# #     if 'user_id' not in session:
# #         return jsonify({'success': False, 'message': 'Please login first'})
    
# #     data = request.get_json()
# #     product_name = data.get('product_name')
# #     product_details = data.get('product_details', '')
# #     image_data = data.get('image_data', '')
    
# #     conn = get_db_connection()
# #     conn.execute('INSERT INTO favorites (user_id, product_name, product_details, image_data) VALUES (?, ?, ?, ?)',
# #                 (session['user_id'], product_name, product_details, image_data))
# #     conn.commit()
# #     conn.close()
    
# #     return jsonify({'success': True, 'message': 'Added to favorites'})

# # @app.route('/profile')
# # def profile():
# #     if 'user_id' not in session:
# #         return redirect(url_for('login'))
    
# #     conn = get_db_connection()
# #     user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
# #     total_searches = conn.execute('SELECT COUNT(*) as count FROM search_history WHERE user_id = ?',
# #                                  (session['user_id'],)).fetchone()['count']
    
# #     total_favorites = conn.execute('SELECT COUNT(*) as count FROM favorites WHERE user_id = ?',
# #                                   (session['user_id'],)).fetchone()['count']
    
# #     total_generated = conn.execute('SELECT COUNT(*) as count FROM generated_images WHERE user_id = ?',
# #                                   (session['user_id'],)).fetchone()['count']
# #     conn.close()
    
# #     return render_template('profile.html', 
# #                          user=user, 
# #                          total_searches=total_searches, 
# #                          total_favorites=total_favorites,
# #                          total_generated=total_generated)

# # # Static files route for images
# # @app.route('/static/<path:filename>')
# # def static_files(filename):
# #     return app.send_static_file(filename)

# # if __name__ == '__main__':
# #     # Initialize database
# #     init_db()
    
# #     # Load model at startup
# #     print("Initializing application...")
# #     load_model()
    
# #     app.run(debug=True, port=5000)



# from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
# from werkzeug.security import generate_password_hash, check_password_hash
# import pandas as pd
# import numpy as np
# import ast
# import os
# from diffusers import StableDiffusionPipeline
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import sqlite3
# from datetime import datetime
# import secrets
# from PIL import Image
# import base64
# from io import BytesIO
# import logging

# app = Flask(__name__)
# app.secret_key = secrets.token_hex(16)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global variables for the ML model
# fashion_pipeline = None
# fashion_df = pd.DataFrame()

# def load_model():
#     """Initialize the ML model and load fashion dataset"""
#     global fashion_pipeline, fashion_df
#     try:
#         # Use a more efficient model loading approach
#         fashion_pipeline = StableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", 
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             use_safetensors=True
#         )
        
#         # Enable memory efficient attention
#         if torch.cuda.is_available():
#             fashion_pipeline = fashion_pipeline.to("cuda")
#             fashion_pipeline.enable_attention_slicing()
#             fashion_pipeline.enable_xformers_memory_efficient_attention()
#         else:
#             fashion_pipeline = fashion_pipeline.to("cpu")
            
#         logger.info("Fashion pipeline loaded successfully")
#     except Exception as e:
#         logger.error(f"Error loading fashion pipeline: {e}")
#         fashion_pipeline = None
    
#     # Load fashion dataset
#     try:
#         fashion_df = pd.read_csv("updated_recommendation.csv")
#         fashion_df['product_attributes'] = fashion_df['product_attributes'].apply(
#             lambda x: ast.literal_eval(x) if pd.notna(x) else {}
#         )
#         logger.info(f"Fashion dataset loaded with {len(fashion_df)} items")
#     except Exception as e:
#         logger.error(f"Error loading fashion dataset: {e}")
#         fashion_df = pd.DataFrame()

# def init_db():
#     conn = sqlite3.connect('fashion_app.db')
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE NOT NULL,
#             email TEXT UNIQUE NOT NULL,
#             password_hash TEXT NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS search_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id INTEGER,
#             search_prompt TEXT NOT NULL,
#             search_type TEXT NOT NULL DEFAULT 'recommendation',
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS favorites (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id INTEGER,
#             product_name TEXT NOT NULL,
#             product_details TEXT,
#             image_data TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS generated_images (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id INTEGER,
#             prompt TEXT NOT NULL,
#             image_data TEXT NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
    
#     conn.commit()
#     conn.close()

# def get_db_connection():
#     conn = sqlite3.connect('fashion_app.db')
#     conn.row_factory = sqlite3.Row
#     return conn

# def get_fashion_recommendations(category, subcategory=None, gender=None, usage=None, color=None, limit=12):
#     """Get fashion recommendations based on filters using cosine similarity"""
#     global fashion_df
    
#     if fashion_df.empty:
#         return []
    
#     try:
#         # Start with the full dataset
#         filtered_df = fashion_df.copy()
        
#         # Apply filters
#         if category and category != 'All':
#             filtered_df = filtered_df[filtered_df['masterCategory'].str.contains(category, case=False, na=False)]
        
#         if subcategory and subcategory != 'All':
#             filtered_df = filtered_df[filtered_df['subCategory'].str.contains(subcategory, case=False, na=False)]
            
#         if gender and gender != 'All':
#             filtered_df = filtered_df[
#                 (filtered_df['gender'].str.contains(gender, case=False, na=False)) |
#                 (filtered_df['gender'].str.contains('Unisex', case=False, na=False))
#             ]
            
#         if usage and usage != 'All':
#             filtered_df = filtered_df[filtered_df['usage'].str.contains(usage, case=False, na=False)]
            
#         if color and color != 'All':
#             filtered_df = filtered_df[filtered_df['baseColour'].str.contains(color, case=False, na=False)]
        
#         if filtered_df.empty:
#             return []
        
#         # Calculate similarity using cosine similarity
#         # Create feature vectors from categorical data
#         feature_cols = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
#         available_cols = [col for col in feature_cols if col in filtered_df.columns]
        
#         if available_cols:
#             # Fill NaN values and convert to string
#             for col in available_cols:
#                 filtered_df[col] = filtered_df[col].fillna('Unknown').astype(str)
            
#             # Create dummy variables for similarity calculation
#             features = pd.get_dummies(filtered_df[available_cols])
            
#             if len(features) > 1:
#                 # Calculate cosine similarity matrix
#                 similarity_matrix = cosine_similarity(features)
#                 # Calculate average similarity score for each item
#                 avg_similarity = similarity_matrix.mean(axis=1)
#                 filtered_df['similarity_score'] = avg_similarity
#             else:
#                 filtered_df['similarity_score'] = 1.0
#         else:
#             filtered_df['similarity_score'] = 1.0
        
#         # Sort by similarity score and get top results
#         top_products = filtered_df.nlargest(limit, 'similarity_score')
        
#         recommendations = []
#         for _, product in top_products.iterrows():
#             recommendations.append({
#                 'name': product.get('productDisplayName', 'Fashion Item'),
#                 'category': product.get('masterCategory', 'Fashion'),
#                 'subcategory': product.get('subCategory', ''),
#                 'article_type': product.get('articleType', ''),
#                 'color': product.get('baseColour', 'Multi'),
#                 'usage': product.get('usage', 'Casual'),
#                 'season': product.get('season', 'All'),
#                 'image': f"images/{product.get('image', 'placeholder.jpg')}",
#                 'similarity_score': float(product.get('similarity_score', 0))
#             })
        
#         return recommendations
        
#     except Exception as e:
#         logger.error(f"Error in get_fashion_recommendations: {e}")
#         return []

# def generate_fashion_image(prompt, num_inference_steps=15):
#     """Generate fashion image using Stable Diffusion with optimized settings"""
#     global fashion_pipeline
    
#     if fashion_pipeline is None:
#         return None, "AI model not loaded. Please try again later."
    
#     try:
#         # Clean and enhance the prompt for better fashion results
#         clean_prompt = prompt.strip()
        
#         # Add fashion-specific keywords for better results
#         enhanced_prompt = f"high quality fashion photography, {clean_prompt}, professional styling, clean background, detailed, 4k resolution"
        
#         # Limit prompt length to avoid issues
#         if len(enhanced_prompt) > 250:
#             enhanced_prompt = f"fashion photography, {clean_prompt}, professional, detailed"
        
#         logger.info(f"Generating image with enhanced prompt: {enhanced_prompt[:100]}...")
        
#         # Generate image with optimized settings for speed and quality
#         with torch.no_grad():
#             # Clear GPU cache if using CUDA
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
            
#             # Generate with reduced steps for faster generation
#             result = fashion_pipeline(
#                 enhanced_prompt,
#                 num_inference_steps=num_inference_steps,  # Reduced from 20 to 15
#                 guidance_scale=7.0,  # Slightly reduced for faster generation
#                 height=512,
#                 width=512,
#                 generator=torch.Generator().manual_seed(42)  # Fixed seed for consistency
#             )
            
#             if not result.images or len(result.images) == 0:
#                 return None, "No image generated. Please try a different prompt."
            
#             image = result.images[0]
        
#         # Convert to base64 for web display
#         buffered = BytesIO()
#         image.save(buffered, format="PNG", optimize=True, quality=85)
#         img_str = base64.b64encode(buffered.getvalue()).decode()
        
#         logger.info("Image generation completed successfully")
#         return img_str, None
        
#     except torch.cuda.OutOfMemoryError:
#         logger.error("GPU out of memory")
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         return None, "GPU memory full. Please try a shorter prompt or try again later."
    
#     except Exception as e:
#         logger.error(f"Error in generate_fashion_image: {e}")
#         error_msg = str(e)
        
#         # Provide user-friendly error messages
#         if "CUDA" in error_msg:
#             return None, "GPU error. Please try again."
#         elif "memory" in error_msg.lower():
#             return None, "Memory error. Please try a shorter prompt."
#         elif "timeout" in error_msg.lower():
#             return None, "Generation timed out. Please try a simpler prompt."
#         else:
#             return None, "Generation failed. Please try a different prompt or try again later."

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         data = request.get_json()
#         username = data.get('username')
#         email = data.get('email')
#         password = data.get('password')
        
#         if not username or not email or not password:
#             return jsonify({'success': False, 'message': 'All fields are required'})
        
#         conn = get_db_connection()
        
#         # Check if user already exists
#         existing_user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
#                                    (username, email)).fetchone()
        
#         if existing_user:
#             conn.close()
#             return jsonify({'success': False, 'message': 'User already exists'})
        
#         # Create new user
#         password_hash = generate_password_hash(password)
#         conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
#                     (username, email, password_hash))
#         conn.commit()
#         conn.close()
        
#         return jsonify({'success': True, 'message': 'Account created successfully'})
    
#     return render_template('auth.html', mode='signup')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         data = request.get_json()
#         username = data.get('username')
#         password = data.get('password')
        
#         if not username or not password:
#             return jsonify({'success': False, 'message': 'Username and password required'})
        
#         conn = get_db_connection()
#         user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
#                           (username, username)).fetchone()
#         conn.close()
        
#         if user and check_password_hash(user['password_hash'], password):
#             session['user_id'] = user['id']
#             session['username'] = user['username']
#             return jsonify({'success': True, 'message': 'Login successful'})
#         else:
#             return jsonify({'success': False, 'message': 'Invalid credentials'})
    
#     return render_template('auth.html', mode='login')

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('home'))

# @app.route('/dashboard')
# def dashboard():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
    
#     # Initialize the model if not already loaded
#     global fashion_pipeline, fashion_df
#     if fashion_pipeline is None and fashion_df.empty:
#         load_model()
    
#     conn = get_db_connection()
#     recent_searches = conn.execute(
#         'SELECT * FROM search_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
#         (session['user_id'],)
#     ).fetchall()
    
#     favorites = conn.execute(
#         'SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
#         (session['user_id'],)
#     ).fetchall()
    
#     generated_images = conn.execute(
#         'SELECT * FROM generated_images WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
#         (session['user_id'],)
#     ).fetchall()
#     conn.close()
    
#     return render_template('dashboard.html', 
#                          recent_searches=recent_searches, 
#                          favorites=favorites,
#                          generated_images=generated_images)

# @app.route('/recommend')
# def recommend():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
    
#     # Get filter options from dataset
#     global fashion_df
#     if fashion_df.empty:
#         load_model()
    
#     categories = ['All'] + sorted(fashion_df['masterCategory'].dropna().unique().tolist())
#     genders = ['All'] + sorted(fashion_df['gender'].dropna().unique().tolist())
#     colors = ['All'] + sorted(fashion_df['baseColour'].dropna().unique().tolist())
#     usages = ['All'] + sorted(fashion_df['usage'].dropna().unique().tolist())
    
#     return render_template('recommend.html', 
#                          categories=categories,
#                          genders=genders, 
#                          colors=colors,
#                          usages=usages)

# @app.route('/generate')
# def generate():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
#     return render_template('generate.html')

# @app.route('/api/get_subcategories')
# def get_subcategories():
#     category = request.args.get('category')
#     global fashion_df
    
#     if category and category != 'All' and not fashion_df.empty:
#         subcategories = fashion_df[
#             fashion_df['masterCategory'].str.contains(category, case=False, na=False)
#         ]['subCategory'].dropna().unique().tolist()
#         subcategories = ['All'] + sorted(subcategories)
#     else:
#         subcategories = ['All']
    
#     return jsonify(subcategories)

# @app.route('/api/search_recommendations', methods=['POST'])
# def search_recommendations():
#     if 'user_id' not in session:
#         return jsonify({'success': False, 'message': 'Please login first'})
    
#     data = request.get_json()
#     category = data.get('category', 'All')
#     subcategory = data.get('subcategory', 'All')
#     gender = data.get('gender', 'All')
#     usage = data.get('usage', 'All')
#     color = data.get('color', 'All')
    
#     # Save search to history
#     search_prompt = f"Category: {category}, Gender: {gender}, Usage: {usage}, Color: {color}"
#     conn = get_db_connection()
#     conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
#                 (session['user_id'], search_prompt, 'recommendation'))
#     conn.commit()
#     conn.close()
    
#     # Get recommendations
#     recommendations = get_fashion_recommendations(category, subcategory, gender, usage, color)
    
#     return jsonify({
#         'success': True, 
#         'recommendations': recommendations
#     })

# @app.route('/api/generate_image', methods=['POST'])
# def generate_image():
#     if 'user_id' not in session:
#         return jsonify({'success': False, 'message': 'Please login first'})
    
#     data = request.get_json()
#     prompt = data.get('prompt', '').strip()
    
#     if not prompt:
#         return jsonify({'success': False, 'message': 'Please enter a prompt'})
    
#     # Generate image
#     image_data, error = generate_fashion_image(prompt)
    
#     if error:
#         return jsonify({'success': False, 'message': f'Error generating image: {error}'})
    
#     # Save to database
#     conn = get_db_connection()
#     conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
#                 (session['user_id'], prompt, 'generation'))
    
#     conn.execute('INSERT INTO generated_images (user_id, prompt, image_data) VALUES (?, ?, ?)',
#                 (session['user_id'], prompt, image_data))
#     conn.commit()
#     conn.close()
    
#     return jsonify({
#         'success': True, 
#         'image': image_data,
#         'prompt': prompt
#     })

# @app.route('/api/add_favorite', methods=['POST'])
# def add_favorite():
#     if 'user_id' not in session:
#         return jsonify({'success': False, 'message': 'Please login first'})
    
#     data = request.get_json()
#     product_name = data.get('product_name')
#     product_details = data.get('product_details', '')
#     image_data = data.get('image_data', '')
    
#     conn = get_db_connection()
#     conn.execute('INSERT INTO favorites (user_id, product_name, product_details, image_data) VALUES (?, ?, ?, ?)',
#                 (session['user_id'], product_name, product_details, image_data))
#     conn.commit()
#     conn.close()
    
#     return jsonify({'success': True, 'message': 'Added to favorites'})

# @app.route('/profile')
# def profile():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
    
#     conn = get_db_connection()
#     user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
#     total_searches = conn.execute('SELECT COUNT(*) as count FROM search_history WHERE user_id = ?',
#                                  (session['user_id'],)).fetchone()['count']
    
#     total_favorites = conn.execute('SELECT COUNT(*) as count FROM favorites WHERE user_id = ?',
#                                   (session['user_id'],)).fetchone()['count']
    
#     total_generated = conn.execute('SELECT COUNT(*) as count FROM generated_images WHERE user_id = ?',
#                                   (session['user_id'],)).fetchone()['count']
#     conn.close()
    
#     return render_template('profile.html', 
#                          user=user, 
#                          total_searches=total_searches, 
#                          total_favorites=total_favorites,
#                          total_generated=total_generated)

# # Static files route for images
# @app.route('/static/<path:filename>')
# def static_files(filename):
#     try:
#         return app.send_static_file(filename)
#     except Exception as e:
#         logger.error(f"Error serving static file {filename}: {e}")
#         # Return a 404 if file not found
#         return "", 404

# if __name__ == '__main__':
#     # Initialize database
#     init_db()
    
#     # Load model at startup
#     print("Initializing application...")
#     load_model()
    
#     app.run(debug=True, port=5000)



from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import ast
import os
from diffusers import StableDiffusionPipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import secrets
from PIL import Image
import base64
from io import BytesIO
import logging

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the ML model
fashion_pipeline = None
fashion_df = pd.DataFrame()

def load_model():
    """Initialize the ML model and load fashion dataset"""
    global fashion_pipeline, fashion_df
    try:
        # Use a more efficient model loading approach
        fashion_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        
        # Enable memory efficient attention
        if torch.cuda.is_available():
            fashion_pipeline = fashion_pipeline.to("cuda")
            fashion_pipeline.enable_attention_slicing()
            fashion_pipeline.enable_xformers_memory_efficient_attention()
        else:
            fashion_pipeline = fashion_pipeline.to("cpu")
            
        logger.info("Fashion pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Error loading fashion pipeline: {e}")
        fashion_pipeline = None
    
    # Load fashion dataset
    try:
        fashion_df = pd.read_csv("updated_recommendation.csv")
        fashion_df['product_attributes'] = fashion_df['product_attributes'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else {}
        )
        logger.info(f"Fashion dataset loaded with {len(fashion_df)} items")
    except Exception as e:
        logger.error(f"Error loading fashion dataset: {e}")
        fashion_df = pd.DataFrame()

def init_db():
    conn = sqlite3.connect('fashion_app.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            search_prompt TEXT NOT NULL,
            search_type TEXT NOT NULL DEFAULT 'recommendation',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product_name TEXT NOT NULL,
            product_details TEXT,
            image_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('fashion_app.db')
    conn.row_factory = sqlite3.Row
    return conn

# def get_fashion_recommendations(category, subcategory=None, gender=None, usage=None, color=None, limit=12):
#     """Get fashion recommendations based on filters using cosine similarity"""
#     global fashion_df
    
#     if fashion_df.empty:
#         return []
    
#     try:
#         # Start with the full dataset
#         filtered_df = fashion_df.copy()
        
#         # Apply filters
#         if category and category != 'All':
#             filtered_df = filtered_df[filtered_df['masterCategory'].str.contains(category, case=False, na=False)]
        
#         if subcategory and subcategory != 'All':
#             filtered_df = filtered_df[filtered_df['subCategory'].str.contains(subcategory, case=False, na=False)]
            
#         if gender and gender != 'All':
#             filtered_df = filtered_df[
#                 (filtered_df['gender'].str.contains(gender, case=False, na=False)) |
#                 (filtered_df['gender'].str.contains('Unisex', case=False, na=False))
#             ]
            
#         if usage and usage != 'All':
#             filtered_df = filtered_df[filtered_df['usage'].str.contains(usage, case=False, na=False)]
            
#         if color and color != 'All':
#             filtered_df = filtered_df[filtered_df['baseColour'].str.contains(color, case=False, na=False)]
        
#         if filtered_df.empty:
#             return []
        
#         # Calculate similarity using cosine similarity
#         # Create feature vectors from categorical data
#         feature_cols = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
#         available_cols = [col for col in feature_cols if col in filtered_df.columns]
        
#         if available_cols:
#             # Fill NaN values and convert to string
#             for col in available_cols:
#                 filtered_df[col] = filtered_df[col].fillna('Unknown').astype(str)
            
#             # Create dummy variables for similarity calculation
#             features = pd.get_dummies(filtered_df[available_cols])
            
#             if len(features) > 1:
#                 # Calculate cosine similarity matrix
#                 similarity_matrix = cosine_similarity(features)
#                 # Calculate average similarity score for each item
#                 avg_similarity = similarity_matrix.mean(axis=1)
#                 filtered_df['similarity_score'] = avg_similarity
#             else:
#                 filtered_df['similarity_score'] = 1.0
#         else:
#             filtered_df['similarity_score'] = 1.0
        
#         # Sort by similarity score and get top results
#         top_products = filtered_df.nlargest(limit, 'similarity_score')
        
#         recommendations = []
#         for _, product in top_products.iterrows():
#             recommendations.append({
#                 'name': product.get('productDisplayName', 'Fashion Item'),
#                 'category': product.get('masterCategory', 'Fashion'),
#                 'subcategory': product.get('subCategory', ''),
#                 'article_type': product.get('articleType', ''),
#                 'color': product.get('baseColour', 'Multi'),
#                 'usage': product.get('usage', 'Casual'),
#                 'season': product.get('season', 'All'),
#                 'image': f"images/{product.get('image', 'placeholder.jpg')}",
#                 'similarity_score': float(product.get('similarity_score', 0))
#             })
        
#         return recommendations
        
#     except Exception as e:
#         logger.error(f"Error in get_fashion_recommendations: {e}")
#         return []

def get_fashion_recommendations(category, subcategory=None, gender=None, usage=None, color=None, limit=12):
    """Get fashion recommendations based on filters using cosine similarity"""
    global fashion_df
    
    if fashion_df.empty:
        return []
    
    try:
        # Start with the full dataset
        filtered_df = fashion_df.copy()
        
        # Apply filters
        if category and category != 'All':
            filtered_df = filtered_df[filtered_df['masterCategory'].str.contains(category, case=False, na=False)]
        
        if subcategory and subcategory != 'All':
            filtered_df = filtered_df[filtered_df['subCategory'].str.contains(subcategory, case=False, na=False)]
            
        if gender and gender != 'All':
            filtered_df = filtered_df[
                (filtered_df['gender'].str.contains(gender, case=False, na=False)) |
                (filtered_df['gender'].str.contains('Unisex', case=False, na=False))
            ]
            
        if usage and usage != 'All':
            filtered_df = filtered_df[filtered_df['usage'].str.contains(usage, case=False, na=False)]
            
        if color and color != 'All':
            filtered_df = filtered_df[filtered_df['baseColour'].str.contains(color, case=False, na=False)]
        
        if filtered_df.empty:
            return []
        
        # Calculate similarity using cosine similarity
        # Create feature vectors from categorical data
        feature_cols = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
        available_cols = [col for col in feature_cols if col in filtered_df.columns]
        
        if available_cols:
            # Fill NaN values and convert to string
            for col in available_cols:
                filtered_df[col] = filtered_df[col].fillna('Unknown').astype(str)
            
            # Create dummy variables for similarity calculation
            features = pd.get_dummies(filtered_df[available_cols])
            
            if len(features) > 1:
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(features)
                # Calculate average similarity score for each item
                avg_similarity = similarity_matrix.mean(axis=1)
                filtered_df['similarity_score'] = avg_similarity
            else:
                filtered_df['similarity_score'] = 1.0
        else:
            filtered_df['similarity_score'] = 1.0
        
        # Sort by similarity score and get top results
        top_products = filtered_df.nlargest(limit, 'similarity_score')
        
        recommendations = []
        for _, product in top_products.iterrows():
            # Get image filename and create proper URL
            image_filename = product.get('image', '')
            if image_filename:
                # Use url_for to generate the correct static file URL
                image_url = url_for('static', filename=f'images/{image_filename}', _external=True)
            else:
                # Fallback to placeholder if no image filename
                image_url = url_for('static', filename='images/placeholder.jpg', _external=True)
            
            recommendations.append({
                'name': product.get('productDisplayName', 'Fashion Item'),
                'category': product.get('masterCategory', 'Fashion'),
                'subcategory': product.get('subCategory', ''),
                'article_type': product.get('articleType', ''),
                'color': product.get('baseColour', 'Multi'),
                'usage': product.get('usage', 'Casual'),
                'season': product.get('season', 'All'),
                'image_url': image_url,  # Changed from 'image' to 'image_url' and using proper URL
                'similarity_score': float(product.get('similarity_score', 0))
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in get_fashion_recommendations: {e}")
        return []
def generate_fashion_image(prompt, num_inference_steps=15):
    """Generate fashion image using Stable Diffusion with optimized settings"""
    global fashion_pipeline
    
    if fashion_pipeline is None:
        return None, "AI model not loaded. Please try again later."
    
    try:
        # Clean and enhance the prompt for better fashion results
        clean_prompt = prompt.strip()
        
        # Add fashion-specific keywords for better results
        enhanced_prompt = f"high quality fashion photography, {clean_prompt}, professional styling, clean background, detailed, 4k resolution"
        
        # Limit prompt length to avoid issues
        if len(enhanced_prompt) > 250:
            enhanced_prompt = f"fashion photography, {clean_prompt}, professional, detailed"
        
        logger.info(f"Generating image with enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Generate image with optimized settings for speed and quality
        with torch.no_grad():
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate with reduced steps for faster generation
            result = fashion_pipeline(
                enhanced_prompt,
                num_inference_steps=num_inference_steps,  # Reduced from 20 to 15
                guidance_scale=7.0,  # Slightly reduced for faster generation
                height=512,
                width=512,
                generator=torch.Generator().manual_seed(42)  # Fixed seed for consistency
            )
            
            if not result.images or len(result.images) == 0:
                return None, "No image generated. Please try a different prompt."
            
            image = result.images[0]
        
        # Convert to base64 for web display
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("Image generation completed successfully")
        return img_str, None
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, "GPU memory full. Please try a shorter prompt or try again later."
    
    except Exception as e:
        logger.error(f"Error in generate_fashion_image: {e}")
        error_msg = str(e)
        
        # Provide user-friendly error messages
        if "CUDA" in error_msg:
            return None, "GPU error. Please try again."
        elif "memory" in error_msg.lower():
            return None, "Memory error. Please try a shorter prompt."
        elif "timeout" in error_msg.lower():
            return None, "Generation timed out. Please try a simpler prompt."
        else:
            return None, "Generation failed. Please try a different prompt or try again later."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        
        # Check if user already exists
        existing_user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
                                   (username, email)).fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'success': False, 'message': 'User already exists'})
        
        # Create new user
        password_hash = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                    (username, email, password_hash))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Account created successfully'})
    
    return render_template('auth.html', mode='signup')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
                          (username, username)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Initialize the model if not already loaded
    global fashion_pipeline, fashion_df
    if fashion_pipeline is None and fashion_df.empty:
        load_model()
    
    conn = get_db_connection()
    recent_searches = conn.execute(
        'SELECT * FROM search_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)
    ).fetchall()
    
    favorites = conn.execute(
        'SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
        (session['user_id'],)
    ).fetchall()
    
    generated_images = conn.execute(
        'SELECT * FROM generated_images WHERE user_id = ? ORDER BY created_at DESC LIMIT 6',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', 
                         recent_searches=recent_searches, 
                         favorites=favorites,
                         generated_images=generated_images)

@app.route('/recommend')
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get filter options from dataset
    global fashion_df
    if fashion_df.empty:
        load_model()
    
    categories = ['All'] + sorted(fashion_df['masterCategory'].dropna().unique().tolist())
    genders = ['All'] + sorted(fashion_df['gender'].dropna().unique().tolist())
    colors = ['All'] + sorted(fashion_df['baseColour'].dropna().unique().tolist())
    usages = ['All'] + sorted(fashion_df['usage'].dropna().unique().tolist())
    
    return render_template('recommend.html', 
                         categories=categories,
                         genders=genders, 
                         colors=colors,
                         usages=usages)

@app.route('/generate')
def generate():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('generate.html')

@app.route('/api/get_subcategories')
def get_subcategories():
    category = request.args.get('category')
    global fashion_df
    
    if category and category != 'All' and not fashion_df.empty:
        subcategories = fashion_df[
            fashion_df['masterCategory'].str.contains(category, case=False, na=False)
        ]['subCategory'].dropna().unique().tolist()
        subcategories = ['All'] + sorted(subcategories)
    else:
        subcategories = ['All']
    
    return jsonify(subcategories)

@app.route('/api/search_recommendations', methods=['POST'])
def search_recommendations():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.get_json()
    category = data.get('category', 'All')
    subcategory = data.get('subcategory', 'All')
    gender = data.get('gender', 'All')
    usage = data.get('usage', 'All')
    color = data.get('color', 'All')
    
    # Save search to history
    search_prompt = f"Category: {category}, Gender: {gender}, Usage: {usage}, Color: {color}"
    conn = get_db_connection()
    conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
                (session['user_id'], search_prompt, 'recommendation'))
    conn.commit()
    conn.close()
    
    # Get recommendations
    recommendations = get_fashion_recommendations(category, subcategory, gender, usage, color)
    
    return jsonify({
        'success': True, 
        'recommendations': recommendations
    })

@app.route('/api/generate_image', methods=['POST'])
def generate_image():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'success': False, 'message': 'Please enter a prompt'})
    
    # Generate image
    image_data, error = generate_fashion_image(prompt)
    
    if error:
        return jsonify({'success': False, 'message': f'Error generating image: {error}'})
    
    # Save to database
    conn = get_db_connection()
    conn.execute('INSERT INTO search_history (user_id, search_prompt, search_type) VALUES (?, ?, ?)',
                (session['user_id'], prompt, 'generation'))
    
    conn.execute('INSERT INTO generated_images (user_id, prompt, image_data) VALUES (?, ?, ?)',
                (session['user_id'], prompt, image_data))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True, 
        'image': image_data,
        'prompt': prompt
    })

@app.route('/api/add_favorite', methods=['POST'])
def add_favorite():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.get_json()
    product_name = data.get('product_name')
    product_details = data.get('product_details', '')
    image_data = data.get('image_data', '')
    
    conn = get_db_connection()
    conn.execute('INSERT INTO favorites (user_id, product_name, product_details, image_data) VALUES (?, ?, ?, ?)',
                (session['user_id'], product_name, product_details, image_data))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Added to favorites'})

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    total_searches = conn.execute('SELECT COUNT(*) as count FROM search_history WHERE user_id = ?',
                                 (session['user_id'],)).fetchone()['count']
    
    total_favorites = conn.execute('SELECT COUNT(*) as count FROM favorites WHERE user_id = ?',
                                  (session['user_id'],)).fetchone()['count']
    
    total_generated = conn.execute('SELECT COUNT(*) as count FROM generated_images WHERE user_id = ?',
                                  (session['user_id'],)).fetchone()['count']
    conn.close()
    
    return render_template('profile.html', 
                         user=user, 
                         total_searches=total_searches, 
                         total_favorites=total_favorites,
                         total_generated=total_generated)

# Debug route to check image availability
@app.route('/debug/images')
def debug_images():
    if 'user_id' not in session:
        return "Please login first"
    
    global fashion_df
    if fashion_df.empty:
        load_model()
    
    # Check first 10 images
    debug_info = []
    for i, (_, row) in enumerate(fashion_df.head(10).iterrows()):
        image_filename = row.get('image', '')
        if image_filename:
            clean_filename = os.path.basename(image_filename)
            image_path = os.path.join('static', 'images', clean_filename)
            exists = os.path.exists(image_path)
            
            debug_info.append({
                'index': i,
                'csv_filename': image_filename,
                'clean_filename': clean_filename,
                'full_path': image_path,
                'exists': exists,
                'product_name': row.get('productDisplayName', 'Unknown')
            })
    
    # Also check what's in the static/images directory
    static_images_path = os.path.join('static', 'images')
    if os.path.exists(static_images_path):
        available_files = os.listdir(static_images_path)[:10]  # First 10 files
    else:
        available_files = ["static/images directory not found"]
    
    return jsonify({
        'csv_images': debug_info,
        'available_files': available_files,
        'static_path_exists': os.path.exists(static_images_path)
    })

# Static files route for images
@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return app.send_static_file(filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        # Return a 404 if file not found
        return "", 404

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load model at startup
    print("Initializing application...")
    load_model()
    
    # Run Flask app on all interfaces, port 8080, debug off
    app.run(host="0.0.0.0", port=8080, debug=False)
