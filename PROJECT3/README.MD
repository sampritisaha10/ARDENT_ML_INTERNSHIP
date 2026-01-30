# Customer Segmentation using K-Means Clustering

A machine learning project demonstrating unsupervised learning through K-Means clustering to segment customers based on their annual income and spending behavior.

## Project Overview

This project uses K-Means clustering algorithm to identify distinct customer segments based on synthetic data representing annual income and spending scores. The visualization helps businesses understand different customer groups for targeted marketing strategies.

## Dataset

The project uses **synthetic customer data** generated using NumPy's random number generation:
- **150 samples** (customers)
- **2 features**: Annual Income and Spending Score
- **3 distinct clusters** representing different customer segments:
  - Low income, low spending (50 samples)
  - Medium income, medium spending (50 samples)
  - High income, medium spending (50 samples)

### Data Generation
```python
np.random.seed(42)
customer_data = np.vstack([
    np.random.normal(loc=[30,30], scale=[5,5], size=(50,2)),  # Cluster 1
    np.random.normal(loc=[60,60], scale=[5,5], size=(50,2)),  # Cluster 2
    np.random.normal(loc=[90,40], scale=[5,5], size=(50,2))   # Cluster 3
])
```

## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib scikit-learn
```

## Project Structure

The notebook follows a systematic clustering workflow:

1. **Data Generation**: Create synthetic customer data with predefined clusters
2. **Initial Visualization**: Plot raw customer data before clustering
3. **K-Means Clustering**: Apply the algorithm with k=3 clusters
4. **Results Visualization**: Display clustered data with centroids

## Key Components

### K-Means Algorithm
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(customer_data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
```

**Parameters:**
- `n_clusters=3`: Number of customer segments to identify
- `random_state=42`: Ensures reproducibility
- `n_init=10`: Number of times the algorithm runs with different initializations

### Cluster Interpretation

The three identified clusters likely represent:

1. **Cluster 0 (Low-Value Customers)**: Low income, low spending
   - Characteristics: Budget-conscious, price-sensitive
   - Strategy: Discount promotions, value offerings

2. **Cluster 1 (Mid-Value Customers)**: Medium income, medium spending
   - Characteristics: Balanced purchasing behavior
   - Strategy: Loyalty programs, personalized offers

3. **Cluster 2 (High-Potential Customers)**: High income, medium spending
   - Characteristics: High earning potential but conservative spending
   - Strategy: Premium product promotion, exclusive experiences

## Visualizations

### Before Clustering
A scatter plot showing all customers without segmentation, making it difficult to identify patterns.

### After Clustering
A color-coded scatter plot where:
- Each color represents a distinct customer segment
- Large 'X' markers indicate cluster centroids (centers)
- Grid lines help identify boundaries between segments

## Usage

1. Open the Jupyter notebook in Google Colab or your local environment
2. Run all cells sequentially
3. Observe the before/after visualizations
4. The final cell displays segmented customers with centroids

## Key Learning Outcomes

- Understanding unsupervised learning concepts
- Implementing K-Means clustering algorithm
- Visualizing high-dimensional data
- Interpreting cluster results for business insights
- Working with synthetic datasets

## Real-World Applications

Customer segmentation has important applications in:
- **Marketing Strategy**: Targeted campaigns for different customer groups
- **Product Development**: Tailoring products to specific segments
- **Resource Allocation**: Focusing efforts on high-value customers
- **Personalization**: Customizing user experiences
- **Retention Programs**: Developing segment-specific loyalty strategies

## Algorithm Details

### How K-Means Works
1. **Initialize**: Randomly place k centroids in the feature space
2. **Assign**: Assign each data point to the nearest centroid
3. **Update**: Recalculate centroid positions based on assigned points
4. **Repeat**: Iterate steps 2-3 until convergence (centroids stop moving)

### Why K=3?
The data was intentionally generated with 3 clusters. In real-world scenarios, you would use techniques like:
- **Elbow Method**: Plot within-cluster sum of squares vs. k
- **Silhouette Score**: Measure cluster cohesion and separation
- **Domain Knowledge**: Business requirements may dictate number of segments

## Extending the Project

Potential improvements and extensions:

1. **Feature Engineering**: Add more customer attributes (age, location, purchase frequency)
2. **Optimal K Selection**: Implement elbow method or silhouette analysis
3. **3D Visualization**: Add a third feature dimension
4. **Real Data**: Use actual customer transaction data
5. **Other Algorithms**: Try DBSCAN, Hierarchical Clustering, or Gaussian Mixture Models
6. **Evaluation Metrics**: Calculate silhouette score, Davies-Bouldin index
7. **Feature Scaling**: Normalize features for better performance
8. **Cluster Profiling**: Generate detailed statistics for each segment

## Limitations

- Synthetic data may not capture real-world complexity
- K-Means assumes spherical clusters of similar size
- Sensitive to initial centroid placement (mitigated by n_init parameter)
- Requires pre-specifying number of clusters
- Outliers can significantly impact results

## Best Practices

1. **Feature Scaling**: Standardize features if they have different scales
2. **Multiple Runs**: Use n_init parameter to avoid local minima
3. **Validation**: Check cluster quality with silhouette scores
4. **Business Context**: Interpret results with domain knowledge
5. **Visualization**: Always visualize results for intuition

## License

This project uses scikit-learn and is intended for educational purposes.

## Acknowledgments

- Libraries: scikit-learn, NumPy, Matplotlib
- Algorithm: K-Means Clustering (MacQueen, 1967)

---

**Note**: This is an educational demonstration of unsupervised learning. Real customer segmentation projects require careful feature selection, data preprocessing, and validation with domain experts.
