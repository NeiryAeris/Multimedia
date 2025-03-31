from model import load_feature_model
from utils import search_image_optimized, build_balltrees, load_or_build_balltrees
from image_loader import load_image
from features import color, deep, shape, texture

# Path to query image
query_image_path = "sample_query1.jpg"  # Replace with your image file

# Load and extract features from query image
img = load_image(query_image_path)
query_features = {
    "color_histogram": color.extract_color_histogram(img),
    "shape_descriptor": shape.extract_shape_descriptor(img),
    "texture_descriptor": texture.extract_texture_descriptor(img),
    "deep_embedding": deep.extract_deep_embedding(img)
}

# Load feature database and build BallTrees
# image_paths, color_array, shape_array, texture_array, deep_array = load_database_features()
# trees = build_balltrees(color_array, shape_array, texture_array, deep_array)
# arrays = (color_array, shape_array, texture_array, deep_array)

trees, arrays, image_paths = load_or_build_balltrees()

# Perform the search
results = search_image_optimized(query_features, trees, arrays, image_paths, top_k_tree=30, top_n_final=5)

# Display results
print("\nTop Matching Images:")
for path, total_sim, sims in results:
    print(f"Image: {path}")
    print(f"  ➤ Total Similarity: {total_sim:.4f}")
    print(f"     - Color:   {sims['color']:.4f}")
    print(f"     - Shape:   {sims['shape']:.4f}")
    print(f"     - Texture: {sims['texture']:.4f}")
    print(f"     - Deep:    {sims['deep']:.4f}")
