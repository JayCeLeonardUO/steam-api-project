import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.dates as mdates
from datetime import datetime


def lm_gamecount(df):
    # Convert release_date to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])

    # Create a numerical representation of dates for modeling
    # Convert dates to ordinal (number of days since 1970-01-01)
    df['date_ordinal'] = df['release_date'].apply(lambda x: x.toordinal())
    min_date = df['date_ordinal'].min()
    df['date_scaled'] = df['date_ordinal'] - min_date  # Days since first game

    # Sort by date and calculate cumulative sum
    df_sorted = df.sort_values('date_ordinal')
    df_sorted['cumulative_count'] = range(1, len(df_sorted) + 1)

    # Resample to monthly frequency for better visualization
    monthly_data = pd.DataFrame({
        'date': df_sorted['release_date'],
        'count': 1
    }).set_index('date').resample('M').count().reset_index()

    monthly_data['cumulative_count'] = monthly_data['count'].cumsum()
    monthly_data['date_ordinal'] = monthly_data['date'].apply(lambda x: x.toordinal())
    monthly_data['date_scaled'] = monthly_data['date_ordinal'] - min_date

    # Create feature matrix X and target vector y
    X = monthly_data[['date_scaled']].values
    y = monthly_data[['cumulative_count']].values

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_linear_pred = linear_model.predict(X)
    y_linear_test_pred = linear_model.predict(X_test)
    linear_r2 = r2_score(y_test, y_linear_test_pred)
    linear_mse = mean_squared_error(y_test, y_linear_test_pred)

    # Quadratic model
    poly_features_2 = PolynomialFeatures(degree=2)
    X_poly_2 = poly_features_2.fit_transform(X)
    X_train_poly_2 = poly_features_2.transform(X_train)
    X_test_poly_2 = poly_features_2.transform(X_test)

    quad_model = LinearRegression()
    quad_model.fit(X_train_poly_2, y_train)
    y_quad_pred = quad_model.predict(X_poly_2)
    y_quad_test_pred = quad_model.predict(X_test_poly_2)
    quad_r2 = r2_score(y_test, y_quad_test_pred)
    quad_mse = mean_squared_error(y_test, y_quad_test_pred)

    # Cubic model
    poly_features_3 = PolynomialFeatures(degree=3)
    X_poly_3 = poly_features_3.fit_transform(X)
    X_train_poly_3 = poly_features_3.transform(X_train)
    X_test_poly_3 = poly_features_3.transform(X_test)

    cubic_model = LinearRegression()
    cubic_model.fit(X_train_poly_3, y_train)
    y_cubic_pred = cubic_model.predict(X_poly_3)
    y_cubic_test_pred = cubic_model.predict(X_test_poly_3)
    cubic_r2 = r2_score(y_test, y_cubic_test_pred)
    cubic_mse = mean_squared_error(y_test, y_cubic_test_pred)

    # Exponential model (log transform)
    log_y = np.log(y)
    log_model = LinearRegression()
    log_model.fit(X_train, np.log(y_train))
    y_log_pred = np.exp(log_model.predict(X))
    y_log_test_pred = np.exp(log_model.predict(X_test))
    log_r2 = r2_score(y_test, y_log_test_pred)
    log_mse = mean_squared_error(y_test, y_log_test_pred)

    # Plotting
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 1, 1)

    # Plot actual data
    plt.scatter(monthly_data['date'], monthly_data['cumulative_count'], 
                s=10, alpha=0.6, color='gray', label='Observed Data')

    # Plot the models
    plt.plot(monthly_data['date'], y_linear_pred, 'r-', 
            linewidth=2, label=f'Linear (R²={linear_r2:.4f}, MSE={linear_mse:.2f})')
    plt.plot(monthly_data['date'], y_quad_pred, 'g-', 
            linewidth=2, label=f'Quadratic (R²={quad_r2:.4f}, MSE={quad_mse:.2f})')
    plt.plot(monthly_data['date'], y_cubic_pred, 'b-', 
            linewidth=2, label=f'Cubic (R²={cubic_r2:.4f}, MSE={cubic_mse:.2f})')
    plt.plot(monthly_data['date'], y_log_pred, 'm-', 
            linewidth=2, label=f'Exponential (R²={log_r2:.4f}, MSE={log_mse:.2f})')

    plt.title('Cumulative Growth of Steam Games with Fitted Models', fontsize=16)
    plt.ylabel('Cumulative Number of Games', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Subplot for residuals
    plt.subplot(2, 1, 2)
    plt.scatter(monthly_data['date'], y - y_linear_pred, color='r', alpha=0.3, label='Linear Residuals')
    plt.scatter(monthly_data['date'], y - y_quad_pred, color='g', alpha=0.3, label='Quadratic Residuals')
    plt.scatter(monthly_data['date'], y - y_cubic_pred, color='b', alpha=0.3, label='Cubic Residuals')
    plt.scatter(monthly_data['date'], y - y_log_pred, color='m', alpha=0.3, label='Exponential Residuals')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Residuals for Each Model', fontsize=16)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('steam_games_growth_models.png', dpi=300)
    plt.show()

    # Print summary statistics
    print("\nModel Performance Summary:")
    print(f"Linear Model:      R² = {linear_r2:.4f}, MSE = {linear_mse:.2f}")
    print(f"Quadratic Model:   R² = {quad_r2:.4f}, MSE = {quad_mse:.2f}")
    print(f"Cubic Model:       R² = {cubic_r2:.4f}, MSE = {cubic_mse:.2f}")
    print(f"Exponential Model: R² = {log_r2:.4f}, MSE = {log_mse:.2f}")

    # Estimate coefficients for interpretability
    print("\nLinear Model Coefficients:")
    print(f"Intercept: {linear_model.intercept_[0]:.2f}")
    print(f"Slope: {linear_model.coef_[0][0]:.4f} games per day")
    print(f"Estimated annual growth rate: {linear_model.coef_[0][0] * 365:.2f} games per year")

    # For quadratic model
    print("\nQuadratic Model Coefficients:")
    print(f"Intercept: {quad_model.intercept_[0]:.2f}")
    print(f"X coefficient: {quad_model.coef_[0][1]:.6f}")
    print(f"X² coefficient: {quad_model.coef_[0][2]:.8f}")

def plot_market_price_graph(df):
    """
    Generate a price density plot with market price point categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing game data with a 'price' column
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The generated plot objects that can be further customized or displayed
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data
    price_df = df[df['price'] > 0].copy()  # Exclude free games
    price_df = price_df[price_df['price'] < 100]  # Limit to reasonable price range
    
    # Create density plot
    sns.kdeplot(data=price_df, x='price', fill=True, cut=0, common_norm=False, ax=ax)
    
    # Set titles and labels
    ax.set_title('Price Density - Market Price Point Competition', fontsize=14)
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Define price categories
    price_categories = [
        (0, 4.99, 'Budget', 'lightblue'),
        (4.99, 14.99, 'Value', 'lightgreen'),
        (14.99, 29.99, 'Standard', 'wheat'),
        (29.99, 59.99, 'Premium', 'salmon'),
        (59.99, 100, 'Deluxe', 'plum')
    ]
    
    # Add category bands
    y_max = ax.get_ylim()[1]
    for start, end, label, color in price_categories:
        ax.axvspan(start, end, alpha=0.2, color=color)
        ax.text((start + end) / 2, y_max * 0.85, label, ha='center', 
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig, ax

def project_steam_games(
    df, 
    target_date=datetime(2026, 1, 1),
    polynomial_degree=3,
    fig_size=(14, 8),
    save_fig=True,
    fig_path='steam_games_projection.png',
    release_date_col='release_date'
):
    """
    Project the future number of Steam games based on historical data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with a release date column
    target_date : datetime, optional
        The future date to predict (default: January 1, 2026)
    polynomial_degree : int, optional
        Degree of the polynomial model (default: 3)
    fig_size : tuple, optional
        Size of the figure (default: (14,.8))
    save_fig : bool, optional
        Whether to save the figure (default: True)
    fig_path : str, optional
        Path to save the figure (default: 'steam_games_projection.png')
    release_date_col : str, optional
        Column name containing release dates (default: 'release_date')
        
    Returns:
    --------
    dict
        Dictionary containing prediction results and model information
    """
    # Handle date formatting
    df = df.copy()
    df[release_date_col] = pd.to_datetime(df[release_date_col], errors='coerce')
    df = df.dropna(subset=[release_date_col])
    
    # Create numerical representation of dates
    df['date_ordinal'] = df[release_date_col].apply(lambda x: x.toordinal())
    min_date = df['date_ordinal'].min()
    df['date_scaled'] = df['date_ordinal'] - min_date  # Days since first game
    
    # Sort by date and calculate cumulative sum
    df_sorted = df.sort_values('date_ordinal')
    df_sorted['cumulative_count'] = range(1, len(df_sorted) + 1)
    
    # Resample to monthly frequency for modeling
    monthly_data = pd.DataFrame({
        'date': df_sorted[release_date_col],
        'count': 1
    }).set_index('date').resample('M').count().reset_index()
    
    monthly_data['cumulative_count'] = monthly_data['count'].cumsum()
    monthly_data['date_ordinal'] = monthly_data['date'].apply(lambda x: x.toordinal())
    monthly_data['date_scaled'] = monthly_data['date_ordinal'] - min_date
    
    # Create feature matrix X and target vector y
    X = monthly_data[['date_scaled']].values
    y = monthly_data[['cumulative_count']].values
    
    # Fit polynomial model
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Create future date
    future_ordinal = target_date.toordinal()
    future_scaled = future_ordinal - min_date
    
    # Transform future date for prediction
    future_X = np.array([[future_scaled]])
    future_X_poly = poly_features.transform(future_X)
    
    # Make prediction
    predicted_games = model.predict(future_X_poly)[0][0]
    
    # Get current date's value for comparison
    current_date = datetime.now()
    current_ordinal = current_date.toordinal()
    current_scaled = current_ordinal - min_date
    current_X = np.array([[current_scaled]])
    current_X_poly = poly_features.transform(current_X)
    current_games = model.predict(current_X_poly)[0][0]
    
    # Calculate the increase
    increase = predicted_games - current_games
    
    # Get the total as of the last date in the data
    last_date = monthly_data['date'].max()
    last_games = monthly_data.loc[monthly_data['date'] == last_date, 'cumulative_count'].values[0]
    
    # Model coefficients
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Print results
    print(f"Polynomial Model (degree {polynomial_degree}) Coefficients:")
    print(f"Intercept: {intercept:.4f}")
    for i, coef in enumerate(coefficients[1:], 1):
        print(f"X^{i} coefficient: {coef:.12f}")
    
    print(f"\nPrediction for {target_date.strftime('%B %d, %Y')}:")
    print(f"Predicted total number of games: {int(predicted_games):,}")
    print(f"Current number of games (as of today): {int(current_games):,}")
    print(f"Last recorded number in dataset (as of {last_date.strftime('%Y-%m-%d')}): {int(last_games):,}")
    print(f"Projected increase from today to {target_date.strftime('%B %d, %Y')}: {int(increase):,} games")
    
    # Create visualization
    future_months = pd.date_range(start=monthly_data['date'].min(), end=target_date, freq='M')
    future_scaled_values = [(date.toordinal() - min_date) for date in future_months]
    future_X_array = np.array([[x] for x in future_scaled_values])
    future_X_poly_array = poly_features.transform(future_X_array)
    future_predictions = model.predict(future_X_poly_array)
    
    future_df = pd.DataFrame({
        'date': future_months,
        'predicted': [val[0] for val in future_predictions]
    })
    
    # Plot
    plt.figure(figsize=fig_size)
    plt.scatter(monthly_data['date'], monthly_data['cumulative_count'], 
                s=10, alpha=0.6, color='blue', label='Historical Data')
    plt.plot(future_df['date'], future_df['predicted'], 'r-', 
             linewidth=2, label=f'Degree {polynomial_degree} Polynomial Model Prediction')
    
    # Highlight the prediction point
    plt.scatter([target_date], [predicted_games], color='red', s=100, 
                zorder=5, label=f'{target_date.strftime("%b %d, %Y")} Prediction: {int(predicted_games):,} games')
    
    plt.axvline(x=current_date, color='green', linestyle='--', alpha=0.7, 
                label=f'Today: {int(current_games):,} games')
    
    plt.title(f'Steam Games Cumulative Growth with Prediction to {target_date.year}', fontsize=16)
    plt.ylabel('Cumulative Number of Games', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(fig_path, dpi=300)
    
    # Return results
    results = {
        'predicted_games': int(predicted_games),
        'current_games': int(current_games),
        'increase': int(increase),
        'last_date': last_date,
        'last_games': int(last_games),
        'model_coefficients': coefficients.tolist(),
        'model_intercept': float(intercept),
        'monthly_data': monthly_data,
        'future_predictions': future_df
    }
    
    return results



def plot_game_reviews_over_time(df, save_path=None, figsize=(16, 10)):
    """
    Create a scatter plot of Steam games showing release date vs number of reviews.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with 'release_date' and 'num_reviews_total' columns
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    figsize : tuple, optional
        Size of the figure (default: (16, 10))
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The generated plot objects
    """
    # Create a copy of the dataframe with just the columns we need
    plot_df = df[['release_date', 'num_reviews_total']].copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    plot_df = plot_df.dropna(subset=['release_date'])
    
    # Fill missing review counts with 0
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    
    # Log-transform the review counts for color mapping
    plot_df['log_reviews'] = np.log1p(plot_df['num_reviews_total'])
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the scatter plot with transparency to show density
    scatter = ax.scatter(
        plot_df['num_reviews_total'], 
        plot_df['release_date'],
        alpha=0.2, 
        s=5, 
        c=plot_df['log_reviews'], 
        cmap='viridis'
    )
    
    # Use log scale for x-axis due to the skewed distribution of reviews
    ax.set_xscale('log')
    
    # Add small value to handle zeros in log scale
    ax.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
    
    # Add labels and title
    ax.set_xlabel('Number of Reviews (log scale)', fontsize=14)
    ax.set_ylabel('Release Date', fontsize=14)
    ax.set_title('Steam Games: Release Date vs. Number of Reviews', fontsize=18)
    
    # Add a colorbar to show the log review count
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log(Number of Reviews + 1)', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig, ax


def analyze_game_reviews_clustering(
    df, 
    k=5, 
    save_dir=None, 
    figsize=(16, 10),
    show_plots=True,
    show_elbow=True,
    k_range=(2, 15)
):
    """
    Analyze Steam games data using clustering based on release date and review counts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with 'release_date' and 'num_reviews_total' columns
    k : int, optional
        Number of clusters for KMeans (default: 5)
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved
    figsize : tuple, optional
        Size of the figures (default: (16, 10))
    show_plots : bool, optional
        Whether to display the plots (default: True)
    show_elbow : bool, optional
        Whether to generate and show the elbow plot (default: True)
    k_range : tuple, optional
        Range of k values to test for elbow method (default: (2, 15))
        
    Returns:
    --------
    dict
        Dictionary containing the analysis results
    """
    # Create a copy of the dataframe with just the columns we need
    plot_df = df[['release_date', 'num_reviews_total']].copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    plot_df = plot_df.dropna(subset=['release_date'])
    
    # Fill missing review counts with 0
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    
    # Convert to numeric for clustering
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Log-transform the review counts for better visualization
    plot_df['log_reviews'] = np.log1p(plot_df['num_reviews_total'])
    
    # 1. Create the initial scatter plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    scatter = ax1.scatter(
        plot_df['num_reviews_total'], 
        plot_df['release_date'],
        alpha=0.2, 
        s=5, 
        c=plot_df['log_reviews'], 
        cmap='viridis'
    )
    
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.yaxis.set_major_locator(mdates.YearLocator(2))
    
    ax1.set_xlabel('Number of Reviews (log scale)', fontsize=14)
    ax1.set_ylabel('Release Date', fontsize=14)
    ax1.set_title('Steam Games: Release Date vs. Number of Reviews', fontsize=18)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Log(Number of Reviews + 1)', fontsize=12)
    
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/steam_games_review_date_scatter.png", dpi=300)
    
    # 2. Prepare data for clustering
    X = plot_df[['release_date_numeric', 'log_reviews']].copy()
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Data for clustering: {len(X)} rows")
    
    # Scale the data using RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Elbow method (optional)
    if show_elbow:
        inertias = []
        min_k, max_k = k_range
        K_range = range(min_k, max_k + 1)
        
        for k_val in K_range:
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(K_range, inertias, 'o-')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('Elbow Method for Optimal k')
        ax_elbow.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/steam_games_kmeans_elbow.png", dpi=300)
    
    # 4. Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add the cluster labels back to our dataframe
    X['cluster'] = labels
    plot_df = plot_df.reset_index().merge(X.reset_index(), left_on='index', right_on='index', how='inner')
    
    # 5. Create the clustered plot
    fig_cluster, ax_cluster = plt.subplots(figsize=figsize)
    
    # Plot each cluster with a different color
    for cluster_id in range(k):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        ax_cluster.scatter(
            cluster_data['num_reviews_total'], 
            cluster_data['release_date'],
            alpha=0.4, 
            s=10, 
            label=f'Cluster {cluster_id+1}'
        )
    
    ax_cluster.set_xscale('log')
    ax_cluster.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax_cluster.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_cluster.yaxis.set_major_locator(mdates.YearLocator(2))
    
    ax_cluster.set_xlabel('Number of Reviews (log scale)', fontsize=14)
    ax_cluster.set_ylabel('Release Date', fontsize=14)
    ax_cluster.set_title(f'Steam Games Clustered by Release Date and Review Count (K={k})', fontsize=18)
    ax_cluster.legend(fontsize=12)
    ax_cluster.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/steam_games_kmeans_clusters.png", dpi=300)
    
    # 6. Analyze the clusters
    cluster_analysis = []
    print("\nCluster Analysis:")
    for cluster_id in range(k):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        min_reviews = cluster_data['num_reviews_total'].min()
        max_reviews = cluster_data['num_reviews_total'].max()
        median_reviews = cluster_data['num_reviews_total'].median()
        avg_year = cluster_data['release_date'].dt.year.mean()
        size = len(cluster_data)
        pct = size / len(plot_df) * 100
        
        # Calculate review range thresholds for this cluster
        q1 = cluster_data['num_reviews_total'].quantile(0.25)
        q3 = cluster_data['num_reviews_total'].quantile(0.75)
        min_year = cluster_data['release_date'].min().year
        max_year = cluster_data['release_date'].max().year
        
        cluster_info = {
            'cluster_id': cluster_id + 1,
            'size': size,
            'percentage': pct,
            'min_year': min_year,
            'max_year': max_year,
            'avg_year': avg_year,
            'median_reviews': median_reviews,
            'q1_reviews': q1,
            'q3_reviews': q3,
            'min_reviews': min_reviews,
            'max_reviews': max_reviews
        }
        
        cluster_analysis.append(cluster_info)
        
        print(f"Cluster {cluster_id+1}: {size} games ({pct:.1f}%)")
        print(f"  - Release years: {min_year} to {max_year} (avg: {avg_year:.1f})")
        print(f"  - Reviews: median={median_reviews:.0f}, Q1={q1:.0f}, Q3={q3:.0f}")
        print(f"  - Review range: {min_reviews:.0f} to {max_reviews:.0f}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    
    # Return analysis results
    results = {
        'raw_data': plot_df,
        'cluster_analysis': cluster_analysis,
        'kmeans_model': kmeans,
        'scaler': scaler,
        'k': k,
        'figures': {
            'scatter': fig1,
            'cluster': fig_cluster,
        }
    }
    
    if show_elbow:
        results['figures']['elbow'] = fig_elbow
        results['inertias'] = inertias
    
    return results

def visualize_game_clusters(df, k=5, figsize=(16, 10), save_path=None):
    """
    Create visualization of game clusters based on release date and review counts,
    and generate statistics on the identified clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with 'release_date' and 'num_reviews_total' columns
    k : int, optional
        Number of clusters for KMeans (default: 5)
    figsize : tuple, optional
        Size of the figures (default: (16, 10))
    save_path : str, optional
        Path to save the cluster figure. If None, figure is not saved
        
    Returns:
    --------
    tuple
        (cluster_fig, stats_fig, cluster_data) where cluster_fig is the matplotlib figure 
        with the cluster visualization, stats_fig is a figure with cluster statistics,
        and cluster_data is a DataFrame with the clustered data
    """
    # Create a copy of the dataframe with just the columns we need
    plot_df = df[['release_date', 'num_reviews_total']].copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    plot_df = plot_df.dropna(subset=['release_date'])
    
    # Fill missing review counts with 0
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    
    # Convert to numeric for clustering
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Log-transform the review counts
    plot_df['log_reviews'] = np.log1p(plot_df['num_reviews_total'])
    
    # Prepare data for clustering
    X = plot_df[['release_date_numeric', 'log_reviews']].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Scale the data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add the cluster labels back to our dataframe
    X['cluster'] = labels
    plot_df = plot_df.reset_index().merge(X.reset_index(), left_on='index', right_on='index', how='inner')
    
    # Create color palette
    colors = sns.color_palette("colorblind", k)
    
    # Create the clustered plot
    cluster_fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster with a different color
    for cluster_id in range(k):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        ax.scatter(
            cluster_data['num_reviews_total'], 
            cluster_data['release_date'],
            alpha=0.5, 
            s=15, 
            label=f'Cluster {cluster_id+1}',
            color=colors[cluster_id]
        )
    
    ax.set_xscale('log')
    ax.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))
    
    ax.set_xlabel('Number of Reviews (log scale)', fontsize=14)
    ax.set_ylabel('Release Date', fontsize=14)
    ax.set_title(f'Steam Games Clustered by Release Date and Review Count (K={k})', fontsize=18)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        cluster_fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Gather cluster statistics
    cluster_stats = []
    
    for cluster_id in range(k):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        
        stats = {
            'cluster': f'Cluster {cluster_id+1}',
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(plot_df) * 100,
            'year_min': cluster_data['release_date'].dt.year.min(),
            'year_max': cluster_data['release_date'].dt.year.max(),
            'year_mean': cluster_data['release_date'].dt.year.mean(),
            'reviews_min': cluster_data['num_reviews_total'].min(),
            'reviews_25%': cluster_data['num_reviews_total'].quantile(0.25),
            'reviews_median': cluster_data['num_reviews_total'].median(),
            'reviews_75%': cluster_data['num_reviews_total'].quantile(0.75),
            'reviews_max': cluster_data['num_reviews_total'].max(),
            'color': colors[cluster_id]
        }
        
        cluster_stats.append(stats)
    
    # Create a statistics visualization
    stats_fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cluster size comparison (pie chart)
    sizes = [stat['size'] for stat in cluster_stats]
    labels = [stat['cluster'] for stat in cluster_stats]
    colors_list = [stat['color'] for stat in cluster_stats]
    
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list, startangle=90)
    axes[0, 0].set_title('Cluster Size Distribution', fontsize=16)
    
    # 2. Review count distribution (box plot)
    box_data = []
    box_labels = []
    
    for cluster_id in range(k):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        # Add small value to handle zeros in log scale
        values = np.log1p(cluster_data['num_reviews_total'].values)
        box_data.append(values)
        box_labels.append(f'Cluster {cluster_id+1}')
    
    axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    axes[0, 1].set_title('Log(Review Count) Distribution by Cluster', fontsize=16)
    axes[0, 1].set_ylabel('Log(Number of Reviews + 1)', fontsize=12)
    
    # Customize box colors
    for patch, color in zip(axes[0, 1].artists, colors_list):
        patch.set_facecolor(color)
    
    # 3. Year distribution by cluster
    year_means = [stat['year_mean'] for stat in cluster_stats]
    year_mins = [stat['year_min'] for stat in cluster_stats]
    year_maxes = [stat['year_max'] for stat in cluster_stats]
    
    x = np.arange(len(labels))
    width = 0.5
    
    axes[1, 0].bar(x, year_means, width, yerr=[(y - min_y) for y, min_y in zip(year_means, year_mins)], 
              color=colors_list, alpha=0.7)
    axes[1, 0].set_ylabel('Average Release Year', fontsize=12)
    axes[1, 0].set_title('Release Year by Cluster', fontsize=16)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    
    # 4. Statistics table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for stat in cluster_stats:
        table_data.append([
            stat['cluster'],
            f"{stat['size']} ({stat['percentage']:.1f}%)",
            f"{stat['year_mean']:.1f} ({stat['year_min']}-{stat['year_max']})",
            f"{stat['reviews_median']:.0f} ({stat['reviews_25%']:.0f}-{stat['reviews_75%']:.0f})"
        ])
    
    table_columns = ['Cluster', 'Size (%)', 'Avg Year (Range)', 'Median Reviews (IQR)']
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=table_columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    axes[1, 1].set_title('Cluster Statistics Summary', fontsize=16)
    
    plt.tight_layout()
    
    # Return both figures and the cluster data
    return cluster_fig, stats_fig, plot_df


def visualize_steam_game_metrics(df, figsize=(16, 12), save_path=None):
    """
    Create visualizations of Steam game metrics:
    1. Density plot by review count buckets
    2. Volume plot of recent reviews over time with total reviews on x-axis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
    figsize : tuple, optional
        Size of the figure (default: (16, 12))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing both visualizations
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Define review count buckets
    review_buckets = [
        (0, 0, 'No Reviews'),
        (1, 9, '1-9'),
        (10, 49, '10-49'),
        (50, 99, '50-99'),
        (100, 499, '100-499'),
        (500, 999, '500-999'),
        (1000, 9999, '1K-10K'),
        (10000, float('inf'), '10K+')
    ]
    
    # Create a new column for review buckets
    def assign_bucket(reviews):
        for low, high, label in review_buckets:
            if low <= reviews <= high:
                return label
        return 'Unknown'
    
    plot_df['review_bucket'] = plot_df['num_reviews_total'].fillna(0).apply(assign_bucket)
    
    # Sort the buckets in a logical order
    bucket_order = [label for _, _, label in review_buckets]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Density by review buckets
    # Count games in each bucket
    bucket_counts = plot_df['review_bucket'].value_counts().reindex(bucket_order)
    
    # Calculate percentages
    total_games = len(plot_df)
    bucket_percentages = (bucket_counts / total_games * 100).round(1)
    
    # Create a colormap
    colors = sns.color_palette("viridis", len(bucket_order))
    
    # Plot density (bar chart with counts)
    bars = ax1.bar(
        bucket_order, 
        bucket_counts,
        color=colors
    )
    
    # Add percentage labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height + 5, 
            f"{bucket_percentages.iloc[i]}%",
            ha='center'
        )
    
    ax1.set_title('Steam Games by Review Count Buckets', fontsize=16)
    ax1.set_xlabel('Number of Reviews', fontsize=14)
    ax1.set_ylabel('Number of Games', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Prepare data for second plot
    # Filter to games with recent review data
    review_df = plot_df.dropna(subset=['num_reviews_recent', 'release_date', 'num_reviews_total'])
    
    # Plot 2: Volume of recent reviews over time with total reviews on x-axis
    # Create a scatter plot with size based on recent reviews and color based on recent reviews
    scatter = ax2.scatter(
        review_df['num_reviews_total'],
        review_df['release_date'],
        s=np.sqrt(review_df['num_reviews_recent']) * 2,  # Scale the points based on recent reviews
        c=np.log1p(review_df['num_reviews_recent']),  # Color based on log of recent reviews
        cmap='plasma',
        alpha=0.7
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Log(Recent Reviews + 1)', fontsize=12)
    
    # Add a legend for the size of points
    sizes = [10, 100, 1000, 10000]
    for size in sizes:
        ax2.scatter([], [], s=np.sqrt(size) * 2, c='gray', alpha=0.7, 
                   label=f'{size:,} Recent Reviews')
    ax2.legend(title="Recent Reviews", loc="upper left")
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, review_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax2.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax2.set_ylabel('Release Date', fontsize=14)
    ax2.set_title('Steam Games: Recent Review Volume Over Time by Total Reviews', fontsize=16)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for some thresholds on x-axis
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= review_df['num_reviews_total'].max():
            ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax2.get_ylim()[0]  # Bottom of the plot
            ax2.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Add a reference line to indicate where total reviews = recent reviews
    max_val = max(review_df['num_reviews_total'].max(), review_df['num_reviews_recent'].max())
    ax2.plot([0, max_val], [ax2.get_ylim()[0], ax2.get_ylim()[0]], 
            linestyle='--', color='red', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_steam_game_metrics2(df, figsize=(16, 16), save_path=None):
    """
    Create visualizations of Steam game metrics:
    1. Scatter plot of total reviews across release dates
    2. Scatter plot of total reviews vs release date colored by price (sized by recent reviews)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 16))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing both visualizations
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Scatter plot of total reviews over time
    scatter1 = ax1.scatter(
        plot_df['num_reviews_total'],
        plot_df['release_date'],
        s=15,  # Fixed size for clarity
        c=np.log1p(plot_df['num_reviews_total']),  # Color by log total reviews
        cmap='viridis',
        alpha=0.7
    )
    
    # Add a colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Log(Total Reviews + 1)', fontsize=12)
    
    # Set log scale for x-axis
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax1.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax1.set_ylabel('Release Date', fontsize=14)
    ax1.set_title('Steam Games: Total Reviews Distribution Over Time', fontsize=16)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add review count thresholds
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= plot_df['num_reviews_total'].max():
            ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax1.get_ylim()[0]  # Bottom of the plot
            ax1.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Plot 2: Scatter plot with price coloring and recent reviews sizing
    # Filter to games with price and recent review data
    review_price_df = plot_df.dropna(subset=['price', 'release_date', 'num_reviews_total'])
    
    # Limit price to below $100 for better color mapping
    price_limit = 100
    review_price_df['price_capped'] = review_price_df['price'].clip(upper=price_limit)
    
    # Create a normalization for price
    norm = Normalize(vmin=0, vmax=price_limit)
    
    # Create the scatter plot
    scatter2 = ax2.scatter(
        review_price_df['num_reviews_total'],
        review_price_df['release_date'],
        s=np.sqrt(review_price_df['num_reviews_recent']) * 2,  # Size by recent reviews
        c=review_price_df['price_capped'],  # Color by price
        cmap='coolwarm',  # Red for expensive, blue for cheap
        alpha=0.7,
        norm=norm
    )
    
    # Add a colorbar for price
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(f'Price ($, capped at ${price_limit})', fontsize=12)
    
    # Add a legend for the size of points
    sizes = [10, 100, 1000, 10000]
    for size in sizes:
        ax2.scatter([], [], s=np.sqrt(size) * 2, c='gray', alpha=0.7, 
                   label=f'{size:,} Recent Reviews')
    ax2.legend(title="Recent Reviews", loc="upper left")
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, review_price_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax2.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax2.set_ylabel('Release Date', fontsize=14)
    ax2.set_title('Steam Games: Price (Color) and Recent Reviews (Size) by Total Reviews Over Time', fontsize=16)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add review count thresholds
    for threshold in review_thresholds:
        if threshold <= review_price_df['num_reviews_total'].max():
            ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax2.get_ylim()[0]  # Bottom of the plot
            ax2.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Add price threshold markers (horizontal lines)
    price_thresholds = [0, 9.99, 19.99, 29.99, 49.99, 59.99]
    
    # Add price annotations in the upper right
    ax2_ylim = ax2.get_ylim()
    ax2_xlim = ax2.get_xlim()
    x_pos = ax2_xlim[1] * 0.9
    
    for i, price in enumerate(price_thresholds):
        y_pos = ax2_ylim[0] + (ax2_ylim[1] - ax2_ylim[0]) * (0.1 + i * 0.05)
        ax2.scatter([x_pos], [y_pos], s=50, c=[price], cmap='coolwarm', norm=norm, edgecolor='black')
        ax2.text(x_pos * 1.05, y_pos, f"${price:.2f}", va='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def visualize_steam_game_metrics_with_boundary(df, figsize=(16, 16), save_path=None):
    """
    Create visualizations of Steam game metrics with a boundary separating
    games with and without recent reviews.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 16))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing both visualizations
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Create numeric version of release date for modeling
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Scatter plot with boundary for games with/without recent reviews
    # Create a binary target: games with at least 1 recent review vs none
    plot_df['has_recent_reviews'] = (plot_df['num_reviews_recent'] > 0).astype(int)
    
    # Filter to reasonable number of reviews for better visualization
    boundary_df = plot_df[plot_df['num_reviews_total'] > 0].copy()
    
    # Transform features for better boundary fitting
    boundary_df['log_reviews'] = np.log1p(boundary_df['num_reviews_total'])
    
    # Prepare data for fitting the boundary
    X = boundary_df[['log_reviews', 'release_date_numeric']].values
    y = boundary_df['has_recent_reviews'].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit a Support Vector Classifier to find the boundary
    svc = SVC(kernel='rbf', gamma=0.5)
    svc.fit(X_scaled, y)
    
    # Create a mesh grid for the decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Get the boundary prediction
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # First, plot the scatter
    scatter1 = ax1.scatter(
        boundary_df['num_reviews_total'],
        boundary_df['release_date'],
        c=boundary_df['has_recent_reviews'],
        cmap='coolwarm',
        alpha=0.7,
        s=15
    )
    
    # Convert the mesh grid back to original scale for plotting
    mesh_scaled = np.c_[xx.ravel(), yy.ravel()]
    mesh_unscaled = scaler.inverse_transform(mesh_scaled)
    xx_unscaled = mesh_unscaled[:, 0].reshape(xx.shape)
    yy_unscaled = mesh_unscaled[:, 1].reshape(yy.shape)
    
    # Convert log reviews back
    xx_unscaled_exp = np.exp(xx_unscaled) - 1
    
    # Convert date ordinals back to datetime for contour plotting
    from datetime import datetime as dt
    date_func = np.vectorize(lambda x: dt.fromordinal(int(x)))
    yy_unscaled_dates = date_func(yy_unscaled)
    
    # Plot the decision boundary
    contour = ax1.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
    
    # Add legend
    legend1 = ax1.legend(['Decision Boundary', 'No Recent Reviews', 'Has Recent Reviews'], 
                        loc='upper left', fontsize=12)
    
    # Set log scale for x-axis
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, boundary_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax1.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax1.set_ylabel('Release Date', fontsize=14)
    ax1.set_title('Steam Games: Boundary Between Games With and Without Recent Reviews', fontsize=16)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Calculate percentages
    games_with_recent = boundary_df['has_recent_reviews'].sum()
    total_games = len(boundary_df)
    pct_with_recent = games_with_recent / total_games * 100
    pct_without_recent = 100 - pct_with_recent
    
    # Add text annotation with percentages
    ax1.text(
        0.05, 0.05, 
        f"Games with recent reviews: {games_with_recent} ({pct_with_recent:.1f}%)\n"
        f"Games without recent reviews: {total_games - games_with_recent} ({pct_without_recent:.1f}%)", 
        transform=ax1.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Plot 2: Scatter plot with price coloring and recent reviews sizing
    # Filter to games with price and review data
    review_price_df = plot_df.dropna(subset=['price', 'release_date', 'num_reviews_total'])
    
    # Limit price to below $100 for better color mapping
    price_limit = 100
    review_price_df['price_capped'] = review_price_df['price'].clip(upper=price_limit)
    
    # Create a normalization for price
    norm = Normalize(vmin=0, vmax=price_limit)
    
    # Create the scatter plot
    scatter2 = ax2.scatter(
        review_price_df['num_reviews_total'],
        review_price_df['release_date'],
        s=np.sqrt(review_price_df['num_reviews_recent'] + 1) * 2,  # Size by recent reviews (+1 to show all points)
        c=review_price_df['price_capped'],  # Color by price
        cmap='coolwarm',  # Red for expensive, blue for cheap
        alpha=0.7,
        norm=norm
    )
    
    # Add a colorbar for price
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(f'Price ($, capped at ${price_limit})', fontsize=12)
    
    # Add a legend for the size of points
    sizes = [0, 10, 100, 1000, 10000]
    for size in sizes:
        ax2.scatter([], [], s=np.sqrt(size + 1) * 2, c='gray', alpha=0.7, 
                   label=f'{size:,} Recent Reviews')
    ax2.legend(title="Recent Reviews", loc="upper left")
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, review_price_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax2.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax2.set_ylabel('Release Date', fontsize=14)
    ax2.set_title('Steam Games: Price (Color) and Recent Reviews (Size) by Total Reviews Over Time', fontsize=16)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add review count thresholds
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= review_price_df['num_reviews_total'].max():
            ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax2.get_ylim()[0]  # Bottom of the plot
            ax2.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Add price threshold markers (horizontal lines)
    price_thresholds = [0, 9.99, 19.99, 29.99, 49.99, 59.99]
    
    # Add price annotations in the upper right
    ax2_ylim = ax2.get_ylim()
    ax2_xlim = ax2.get_xlim()
    x_pos = ax2_xlim[1] * 0.9
    
    for i, price in enumerate(price_thresholds):
        y_pos = ax2_ylim[0] + (ax2_ylim[1] - ax2_ylim[0]) * (0.1 + i * 0.05)
        ax2.scatter([x_pos], [y_pos], s=50, c=[price], cmap='coolwarm', norm=norm, edgecolor='black')
        ax2.text(x_pos * 1.05, y_pos, f"${price:.2f}", va='center')
    
    # Add the boundary to the second plot as well
    try:
        ax2.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
        ax2.text(0.05, 0.05, "Boundary: Games with/without recent reviews", 
                transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    except:
        # If boundary plotting fails, skip it
        pass
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def visualize_steam_game_metrics_with_boundary(df, figsize=(16, 16), save_path=None):
    """
    Create visualizations of Steam game metrics with a boundary separating
    games with and without recent reviews.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 16))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing both visualizations
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Create numeric version of release date for modeling
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Scatter plot with boundary for games with/without recent reviews
    # Create a binary target: games with at least 1 recent review vs none
    plot_df['has_recent_reviews'] = (plot_df['num_reviews_recent'] > 0).astype(int)
    
    # Filter to reasonable number of reviews for better visualization
    boundary_df = plot_df[plot_df['num_reviews_total'] > 0].copy()
    
    # Transform features for better boundary fitting
    boundary_df['log_reviews'] = np.log1p(boundary_df['num_reviews_total'])
    
    # Prepare data for fitting the boundary
    X = boundary_df[['log_reviews', 'release_date_numeric']].values
    y = boundary_df['has_recent_reviews'].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit a Support Vector Classifier to find the boundary
    svc = SVC(kernel='rbf', gamma=0.5)
    svc.fit(X_scaled, y)
    
    # Create a mesh grid for the decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Get the boundary prediction
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # First, plot the scatter
    scatter1 = ax1.scatter(
        boundary_df['num_reviews_total'],
        boundary_df['release_date'],
        c=boundary_df['has_recent_reviews'],
        cmap='coolwarm',
        alpha=0.7,
        s=15
    )
    
    # Convert the mesh grid back to original scale for plotting
    mesh_scaled = np.c_[xx.ravel(), yy.ravel()]
    mesh_unscaled = scaler.inverse_transform(mesh_scaled)
    xx_unscaled = mesh_unscaled[:, 0].reshape(xx.shape)
    yy_unscaled = mesh_unscaled[:, 1].reshape(yy.shape)
    
    # Convert log reviews back
    xx_unscaled_exp = np.exp(xx_unscaled) - 1
    
    # Convert date ordinals back to datetime for contour plotting
    from datetime import datetime as dt
    date_func = np.vectorize(lambda x: dt.fromordinal(int(x)))
    yy_unscaled_dates = date_func(yy_unscaled)
    
    # Plot the decision boundary
    contour = ax1.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
    
    # Add legend
    legend1 = ax1.legend(['Decision Boundary', 'No Recent Reviews', 'Has Recent Reviews'], 
                        loc='upper left', fontsize=12)
    
    # Set log scale for x-axis
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, boundary_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax1.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax1.set_ylabel('Release Date', fontsize=14)
    ax1.set_title('Steam Games: Boundary Between Games With and Without Recent Reviews', fontsize=16)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Calculate percentages
    games_with_recent = boundary_df['has_recent_reviews'].sum()
    total_games = len(boundary_df)
    pct_with_recent = games_with_recent / total_games * 100
    pct_without_recent = 100 - pct_with_recent
    
    # Add text annotation with percentages
    ax1.text(
        0.05, 0.05, 
        f"Games with recent reviews: {games_with_recent} ({pct_with_recent:.1f}%)\n"
        f"Games without recent reviews: {total_games - games_with_recent} ({pct_without_recent:.1f}%)", 
        transform=ax1.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Plot 2: Scatter plot with price coloring and recent reviews sizing
    # Filter to games with price and review data
    review_price_df = plot_df.dropna(subset=['price', 'release_date', 'num_reviews_total'])
    
    # Limit price to below $100 for better color mapping
    price_limit = 100
    review_price_df['price_capped'] = review_price_df['price'].clip(upper=price_limit)
    
    # Create a normalization for price
    norm = Normalize(vmin=0, vmax=price_limit)
    
    # Create the scatter plot
    scatter2 = ax2.scatter(
        review_price_df['num_reviews_total'],
        review_price_df['release_date'],
        s=np.sqrt(review_price_df['num_reviews_recent'] + 1) * 2,  # Size by recent reviews (+1 to show all points)
        c=review_price_df['price_capped'],  # Color by price
        cmap='coolwarm',  # Red for expensive, blue for cheap
        alpha=0.7,
        norm=norm
    )
    
    # Add a colorbar for price
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(f'Price ($, capped at ${price_limit})', fontsize=12)
    
    # Add a legend for the size of points
    sizes = [0, 10, 100, 1000, 10000]
    for size in sizes:
        ax2.scatter([], [], s=np.sqrt(size + 1) * 2, c='gray', alpha=0.7, 
                   label=f'{size:,} Recent Reviews')
    ax2.legend(title="Recent Reviews", loc="upper left")
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, review_price_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax2.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax2.set_ylabel('Release Date', fontsize=14)
    ax2.set_title('Steam Games: Price (Color) and Recent Reviews (Size) by Total Reviews Over Time', fontsize=16)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add review count thresholds
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= review_price_df['num_reviews_total'].max():
            ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax2.get_ylim()[0]  # Bottom of the plot
            ax2.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Add price threshold markers (horizontal lines)
    price_thresholds = [0, 9.99, 19.99, 29.99, 49.99, 59.99]
    
    # Add price annotations in the upper right
    ax2_ylim = ax2.get_ylim()
    ax2_xlim = ax2.get_xlim()
    x_pos = ax2_xlim[1] * 0.9
    
    for i, price in enumerate(price_thresholds):
        y_pos = ax2_ylim[0] + (ax2_ylim[1] - ax2_ylim[0]) * (0.1 + i * 0.05)
        ax2.scatter([x_pos], [y_pos], s=50, c=[price], cmap='coolwarm', norm=norm, edgecolor='black')
        ax2.text(x_pos * 1.05, y_pos, f"${price:.2f}", va='center')
    
    # Add the boundary to the second plot as well
    try:
        ax2.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
        ax2.text(0.05, 0.05, "Boundary: Games with/without recent reviews", 
                transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    except:
        # If boundary plotting fails, skip it
        pass
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_steam_game_metrics_with_boundary(df, figsize=(16, 16), save_path=None):
    """
    Create visualizations of Steam game metrics with a boundary separating
    games with and without recent reviews.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 16))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing both visualizations
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Create numeric version of release date for modeling
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Scatter plot with boundary for games with/without recent reviews
    # Create a binary target: games with at least 1 recent review vs none
    plot_df['has_recent_reviews'] = (plot_df['num_reviews_recent'] > 0).astype(int)
    
    # Filter to reasonable number of reviews for better visualization
    boundary_df = plot_df[plot_df['num_reviews_total'] > 0].copy()
    
    # Transform features for better boundary fitting
    boundary_df['log_reviews'] = np.log1p(boundary_df['num_reviews_total'])
    
    # Prepare data for fitting the boundary
    X = boundary_df[['log_reviews', 'release_date_numeric']].values
    y = boundary_df['has_recent_reviews'].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit a Support Vector Classifier to find the boundary
    svc = SVC(kernel='rbf', gamma=0.5)
    svc.fit(X_scaled, y)
    
    # Create a mesh grid for the decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Get the boundary prediction
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # First, plot the scatter
    scatter1 = ax1.scatter(
        boundary_df['num_reviews_total'],
        boundary_df['release_date'],
        c=boundary_df['has_recent_reviews'],
        cmap='coolwarm',
        alpha=0.7,
        s=15
    )
    
    # Convert the mesh grid back to original scale for plotting
    mesh_scaled = np.c_[xx.ravel(), yy.ravel()]
    mesh_unscaled = scaler.inverse_transform(mesh_scaled)
    xx_unscaled = mesh_unscaled[:, 0].reshape(xx.shape)
    yy_unscaled = mesh_unscaled[:, 1].reshape(yy.shape)
    
    # Convert log reviews back
    xx_unscaled_exp = np.exp(xx_unscaled) - 1
    
    # Convert date ordinals back to datetime for contour plotting
    from datetime import datetime as dt
    date_func = np.vectorize(lambda x: dt.fromordinal(int(x)))
    yy_unscaled_dates = date_func(yy_unscaled)
    
    # Plot the decision boundary
    contour = ax1.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
    
    # Add legend
    legend1 = ax1.legend(['Decision Boundary', 'No Recent Reviews', 'Has Recent Reviews'], 
                        loc='upper left', fontsize=12)
    
    # Set log scale for x-axis
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, boundary_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax1.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax1.set_ylabel('Release Date', fontsize=14)
    ax1.set_title('Steam Games: Boundary Between Games With and Without Recent Reviews', fontsize=16)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Calculate percentages
    games_with_recent = boundary_df['has_recent_reviews'].sum()
    total_games = len(boundary_df)
    pct_with_recent = games_with_recent / total_games * 100
    pct_without_recent = 100 - pct_with_recent
    
    # Add text annotation with percentages
    ax1.text(
        0.05, 0.05, 
        f"Games with recent reviews: {games_with_recent} ({pct_with_recent:.1f}%)\n"
        f"Games without recent reviews: {total_games - games_with_recent} ({pct_without_recent:.1f}%)", 
        transform=ax1.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Plot 2: Scatter plot with price coloring and recent reviews sizing
    # Filter to games with price and review data
    review_price_df = plot_df.dropna(subset=['price', 'release_date', 'num_reviews_total'])
    
    # Limit price to below $100 for better color mapping
    price_limit = 100
    review_price_df['price_capped'] = review_price_df['price'].clip(upper=price_limit)
    
    # Create a normalization for price
    norm = Normalize(vmin=0, vmax=price_limit)
    
    # Create the scatter plot
    scatter2 = ax2.scatter(
        review_price_df['num_reviews_total'],
        review_price_df['release_date'],
        s=np.sqrt(review_price_df['num_reviews_recent'] + 1) * 2,  # Size by recent reviews (+1 to show all points)
        c=review_price_df['price_capped'],  # Color by price
        cmap='coolwarm',  # Red for expensive, blue for cheap
        alpha=0.7,
        norm=norm
    )
    
    # Add a colorbar for price
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(f'Price ($, capped at ${price_limit})', fontsize=12)
    
    # Add a legend for the size of points
    sizes = [0, 10, 100, 1000, 10000]
    for size in sizes:
        ax2.scatter([], [], s=np.sqrt(size + 1) * 2, c='gray', alpha=0.7, 
                   label=f'{size:,} Recent Reviews')
    ax2.legend(title="Recent Reviews", loc="upper left")
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlim(0.5, review_price_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax2.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax2.set_ylabel('Release Date', fontsize=14)
    ax2.set_title('Steam Games: Price (Color) and Recent Reviews (Size) by Total Reviews Over Time', fontsize=16)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add review count thresholds
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= review_price_df['num_reviews_total'].max():
            ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            y_pos = ax2.get_ylim()[0]  # Bottom of the plot
            ax2.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Add price threshold markers (horizontal lines)
    price_thresholds = [0, 9.99, 19.99, 29.99, 49.99, 59.99]
    
    # Add price annotations in the upper right
    ax2_ylim = ax2.get_ylim()
    ax2_xlim = ax2.get_xlim()
    x_pos = ax2_xlim[1] * 0.9
    
    for i, price in enumerate(price_thresholds):
        y_pos = ax2_ylim[0] + (ax2_ylim[1] - ax2_ylim[0]) * (0.1 + i * 0.05)
        ax2.scatter([x_pos], [y_pos], s=50, c=[price], cmap='coolwarm', norm=norm, edgecolor='black')
        ax2.text(x_pos * 1.05, y_pos, f"${price:.2f}", va='center')
    
    # Add the boundary to the second plot as well
    try:
        ax2.contour(xx_unscaled_exp, yy_unscaled_dates, Z, levels=[0], colors='black', linestyles='--')
        ax2.text(0.05, 0.05, "Boundary: Games with/without recent reviews", 
                transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    except:
        # If boundary plotting fails, skip it
        pass
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

def visualize_steam_games_with_svm_boundaries(df, figsize=(20, 16), save_path=None):
    """
    Create visualization of Steam games with SVM decision boundaries at 10, 100, and 1000 recent reviews.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (20, 16))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig : matplotlib figure object
        The figure containing the visualization
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Create numeric version of release date for modeling
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Define the thresholds for recent reviews
    review_thresholds = [10, 100, 1000]
    threshold_colors = ['#2ca02c', '#d62728', '#9467bd']  # Green, Red, Purple
    threshold_labels = ['10+ Recent Reviews', '100+ Recent Reviews', '1000+ Recent Reviews']
    
    # Create a review bracket column for coloring
    def categorize_recent_reviews(reviews):
        if reviews == 0:
            return "No Recent Reviews"
        elif reviews < 10:
            return "1-9 Recent Reviews"
        elif reviews < 100:
            return "10-99 Recent Reviews"
        elif reviews < 1000:
            return "100-999 Recent Reviews"
        else:
            return "1000+ Recent Reviews"
    
    plot_df['review_category'] = plot_df['num_reviews_recent'].apply(categorize_recent_reviews)
    
    # Define color mapping for categories
    category_colors = {
        "No Recent Reviews": "#d9d9d9",  # Gray
        "1-9 Recent Reviews": "#66c2a5",  # Teal
        "10-99 Recent Reviews": "#fc8d62",  # Orange
        "100-999 Recent Reviews": "#8da0cb",  # Blue
        "1000+ Recent Reviews": "#e78ac3"   # Pink
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create binary targets for each threshold
    for i, threshold in enumerate(review_thresholds):
        plot_df[f'over_{threshold}_recent'] = (plot_df['num_reviews_recent'] >= threshold).astype(int)
    
    # Filter to games with total reviews above 0
    boundary_df = plot_df[plot_df['num_reviews_total'] > 0].copy()
    
    # Transform features for better boundary fitting
    boundary_df['log_total_reviews'] = np.log1p(boundary_df['num_reviews_total'])
    
    # Fit SVM boundaries for each threshold
    boundary_curves = []
    
    for i, threshold in enumerate(review_thresholds):
        # Prepare data for fitting the boundary
        X = boundary_df[['log_total_reviews', 'release_date_numeric']].values
        y = boundary_df[f'over_{threshold}_recent'].values
        
        # Skip if not enough samples in both classes
        if np.unique(y).size < 2 or np.min(np.bincount(y)) < 10:
            print(f"Skipping threshold {threshold} - not enough samples in both classes")
            continue
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit a Support Vector Classifier
        svc = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
        svc.fit(X_scaled, y)
        
        # Create a mesh grid for the decision boundary
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                            np.arange(y_min, y_max, 0.05))
        
        # Get the boundary prediction
        Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Convert the mesh grid back to original scale for plotting
        mesh_scaled = np.c_[xx.ravel(), yy.ravel()]
        mesh_unscaled = scaler.inverse_transform(mesh_scaled)
        xx_unscaled = mesh_unscaled[:, 0].reshape(xx.shape)
        yy_unscaled = mesh_unscaled[:, 1].reshape(yy.shape)
        
        # Convert log reviews back
        xx_unscaled_exp = np.exp(xx_unscaled) - 1
        
        # Convert date ordinals back to datetime for contour plotting
        from datetime import datetime as dt
        date_func = np.vectorize(lambda x: dt.fromordinal(int(x)))
        yy_unscaled_dates = date_func(yy_unscaled)
        
        # Store boundary data for plotting
        boundary_curves.append((xx_unscaled_exp, yy_unscaled_dates, Z, threshold_colors[i], threshold_labels[i]))
        
        # Calculate stats
        count_over = boundary_df[f'over_{threshold}_recent'].sum()
        pct_over = count_over / len(boundary_df) * 100
        print(f"{threshold}+ recent reviews: {count_over} games ({pct_over:.1f}%)")
    
    # Create the scatter plot
    for category, color in category_colors.items():
        category_data = plot_df[plot_df['review_category'] == category]
        if len(category_data) > 0:
            ax.scatter(
                category_data['num_reviews_total'],
                category_data['release_date'],
                s=np.sqrt(category_data['num_reviews_recent'] + 1) * 2,  # Size by recent reviews
                c=color,
                label=f"{category} ({len(category_data)} games)",
                alpha=0.6
            )
    
    # Plot SVM boundaries with high visibility
    for xx, yy, Z, color, label in boundary_curves:
        contour = ax.contour(xx, yy, Z, levels=[0], colors=[color], linewidths=3)
        
        # Add label at approximate middle of the boundary
        y_mid_idx = Z.shape[0] // 2
        boundary_points = np.where(np.abs(Z[y_mid_idx]) < 0.5)[0]
        
        if len(boundary_points) > 0:
            x_idx = boundary_points[len(boundary_points) // 2]
            x_label_pos = xx[y_mid_idx, x_idx]
            y_label_pos = yy[y_mid_idx, x_idx]
            
            ax.annotate(
                label,
                xy=(x_label_pos, y_label_pos),
                xytext=(x_label_pos * 0.7, y_label_pos),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                fontsize=12,
                fontweight='bold',
                color=color
            )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax.set_xlabel('Total Reviews (log scale)', fontsize=16)
    ax.set_ylabel('Release Date', fontsize=16)
    ax.set_title('Steam Games: SVM Decision Boundaries for Recent Review Activity', fontsize=20)
    
    # Add a legend
    ax.legend(loc='upper left', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for total review count reference
    total_review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in total_review_thresholds:
        if threshold <= plot_df['num_reviews_total'].max():
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.4)
            y_pos = ax.get_ylim()[0]  # Bottom of the plot
            ax.text(threshold*1.1, y_pos, f"{threshold:,}", 
                   fontsize=10, rotation=90, verticalalignment='bottom')
    
    # Create a custom legend for the point sizes
    size_examples = [0, 10, 100, 1000]
    size_elements = []
    for size in size_examples:
        size_elements.append(Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='gray', markersize=np.sqrt(size + 1) * 0.8,
                                   label=f"{size:,} Recent Reviews"))
    
    # Add the second legend for sizes
    size_legend = plt.legend(handles=size_elements, loc='lower left', 
                           title="Point Size Reference", fontsize=12)
    ax.add_artist(size_legend)
    
    # Calculate statistics for each category
    stats_text = "Recent Review Activity:\n"
    for category in category_colors.keys():
        count = len(plot_df[plot_df['review_category'] == category])
        percentage = count / len(plot_df) * 100
        stats_text += f"{category}: {count:,} games ({percentage:.1f}%)\n"
    
    # Add statistics in a text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def run_svm_cross_validation_experiment(df, n_splits=5, save_path=None):
    """
    Run a cross-validation experiment for SVM models at different recent review thresholds.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    save_path : str, optional
        Directory to save results. If None, results are not saved
        
    Returns:
    --------
    dict
        Dictionary containing all experiment results
    """
    # Set the style
    plt.style.use('ggplot')
    sns.set_palette("deep")
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    
    # Create numeric version of release date for modeling
    plot_df['release_date_numeric'] = plot_df['release_date'].apply(lambda x: x.toordinal())
    
    # Define the thresholds for recent reviews
    review_thresholds = [10, 100, 1000]
    threshold_colors = ['#2ca02c', '#d62728', '#9467bd']  # Green, Red, Purple
    threshold_labels = ['10+ Recent Reviews', '100+ Recent Reviews', '1000+ Recent Reviews']
    
    # Filter to games with total reviews above 0
    model_df = plot_df[plot_df['num_reviews_total'] > 0].copy()
    
    # Transform features for better model fitting
    model_df['log_total_reviews'] = np.log1p(model_df['num_reviews_total'])
    
    # Create binary targets for each threshold
    for threshold in review_thresholds:
        model_df[f'over_{threshold}_recent'] = (model_df['num_reviews_recent'] >= threshold).astype(int)
    
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'class_weight': ['balanced', None]
    }
    
    # Store results for each threshold
    results = {}
    
    # Create a figure for ROC curves
    fig_roc, ax_roc = plt.subplots(1, len(review_thresholds), figsize=(20, 6))
    
    # Create a figure for confusion matrices
    fig_cm, ax_cm = plt.subplots(1, len(review_thresholds), figsize=(20, 6))
    
    # Create a figure for decision boundaries
    fig_boundary, ax_boundary = plt.subplots(1, len(review_thresholds), figsize=(20, 6))
    
    # Run experiment for each threshold
    for i, threshold in enumerate(review_thresholds):
        print(f"\nRunning cross-validation for {threshold}+ recent reviews threshold")
        
        # Check class balance
        target = f'over_{threshold}_recent'
        positive_count = model_df[target].sum()
        total_count = len(model_df)
        positive_pct = positive_count / total_count * 100
        
        print(f"Class balance: {positive_count} games with {threshold}+ recent reviews ({positive_pct:.1f}%)")
        
        # Skip if not enough samples in positive class
        if positive_count < 30:
            print(f"Skipping threshold {threshold} - not enough positive samples")
            continue
        
        # Prepare data for cross-validation
        X = model_df[['log_total_reviews', 'release_date_numeric']].values
        y = model_df[target].values
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            SVC(kernel='rbf', probability=True),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run grid search
        grid_search.fit(X_scaled, y)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        # Store cross-validation results
        cv_results = []
        roc_curves = []
        cms = []
        
        # Run detailed cross-validation with best model to get metrics per fold
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone the best model
            model = SVC(kernel='rbf', probability=True, **best_params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            cv_results.append(metrics)
            
            # Store ROC curve data
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves.append((fpr, tpr))
            
            # Store confusion matrix
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            cms.append(cm)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            values = [result[metric] for result in cv_results]
            avg_metrics[metric] = sum(values) / len(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        # Train a final model on all data for visualization
        final_model = SVC(kernel='rbf', probability=True, **best_params)
        final_model.fit(X_scaled, y)
        
        # Plot ROC curves
        ax_roc[i].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        
        for fold, (fpr, tpr) in enumerate(roc_curves):
            ax_roc[i].plot(fpr, tpr, alpha=0.7, label=f'Fold {fold+1}')
        
        ax_roc[i].set_xlabel('False Positive Rate')
        ax_roc[i].set_ylabel('True Positive Rate')
        ax_roc[i].set_title(f'ROC Curves for {threshold}+ Recent Reviews\nAUC: {avg_metrics["roc_auc"]:.3f} ± {avg_metrics["roc_auc_std"]:.3f}')
        ax_roc[i].legend()
        
        # Plot average confusion matrix
        avg_cm = sum(cms) / len(cms)
        ConfusionMatrixDisplay(avg_cm, display_labels=['< Threshold', '≥ Threshold']).plot(
            ax=ax_cm[i], cmap='Blues', values_format='.2f'
        )
        ax_cm[i].set_title(f'Confusion Matrix for {threshold}+ Recent Reviews')
        
        # Plot decision boundary
        # Create a mesh grid for visualization
        x_min, x_max = np.min(X_scaled[:, 0]) - 1, np.max(X_scaled[:, 0]) + 1
        y_min, y_max = np.min(X_scaled[:, 1]) - 1, np.max(X_scaled[:, 1]) + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        # Make predictions on the grid
        Z = final_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Convert back to original scale for plotting
        xx_orig = np.exp(scaler.inverse_transform(np.c_[xx.ravel(), np.zeros(xx.ravel().shape)])[:, 0]) - 1
        xx_orig = xx_orig.reshape(xx.shape)
        
        # Convert date ordinals back to dates
        yy_orig = scaler.inverse_transform(np.c_[np.zeros(yy.ravel().shape), yy.ravel()])[:, 1].reshape(yy.shape)
        yy_dates = np.vectorize(lambda x: datetime.fromordinal(int(x)))(yy_orig)
        
        # Random sample for scatter plot to avoid overcrowding
        sample_size = min(1000, len(model_df))
        sample_idx = np.random.choice(len(model_df), sample_size, replace=False)
        sample_df = model_df.iloc[sample_idx]
        
        # Scatter plot of the data points
        scatter = ax_boundary[i].scatter(
            sample_df['num_reviews_total'],
            sample_df['release_date'],
            c=sample_df[target],
            cmap='coolwarm',
            alpha=0.6,
            s=15
        )
        
        # Plot decision boundary
        ax_boundary[i].contour(xx_orig, yy_dates, Z, levels=[0], colors=[threshold_colors[i]], linewidths=3)
        ax_boundary[i].set_xscale('log')
        ax_boundary[i].set_title(f'Decision Boundary for {threshold}+ Recent Reviews\nF1: {avg_metrics["f1"]:.3f} ± {avg_metrics["f1_std"]:.3f}')
        ax_boundary[i].set_xlabel('Total Reviews (log scale)')
        ax_boundary[i].yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        if i == 0:
            ax_boundary[i].set_ylabel('Release Date')
        
        # Store results
        results[threshold] = {
            'best_params': best_params,
            'cv_results': cv_results,
            'avg_metrics': avg_metrics,
            'final_model': final_model,
            'scaler': scaler
        }
    
    # Adjust layout and save figures
    fig_roc.tight_layout()
    fig_cm.tight_layout()
    fig_boundary.tight_layout()
    
    if save_path:
        fig_roc.savefig(f"{save_path}/roc_curves.png", dpi=300, bbox_inches='tight')
        fig_cm.savefig(f"{save_path}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        fig_boundary.savefig(f"{save_path}/decision_boundaries.png", dpi=300, bbox_inches='tight')
        
        # Create a summary report
        summary_df = pd.DataFrame([
            {
                'threshold': threshold,
                'positive_samples': model_df[f'over_{threshold}_recent'].sum(),
                'positive_pct': model_df[f'over_{threshold}_recent'].mean() * 100,
                'best_C': results[threshold]['best_params']['C'],
                'best_gamma': results[threshold]['best_params']['gamma'],
                'best_class_weight': results[threshold]['best_params']['class_weight'],
                'accuracy': results[threshold]['avg_metrics']['accuracy'],
                'precision': results[threshold]['avg_metrics']['precision'],
                'recall': results[threshold]['avg_metrics']['recall'],
                'f1': results[threshold]['avg_metrics']['f1'],
                'roc_auc': results[threshold]['avg_metrics']['roc_auc']
            }
            for threshold in results.keys()
        ])
        
        summary_df.to_csv(f"{save_path}/svm_summary.csv", index=False)
    
    # Create a figure with feature importance analysis
    if results:
        fig_imp, ax_imp = plt.subplots(1, len(results), figsize=(20, 6))
        
        for i, (threshold, res) in enumerate(results.items()):
            # Create a feature importance grid
            x_values = np.linspace(-3, 3, 20)  # standardized log review values
            y_values = np.linspace(-3, 3, 20)  # standardized date values
            
            importance_grid = np.zeros((len(y_values), len(x_values)))
            
            for x_idx, x_val in enumerate(x_values):
                for y_idx, y_val in enumerate(y_values):
                    # Create a point
                    point = np.array([[x_val, y_val]])
                    
                    # Get decision function value (distance from hyperplane)
                    decision_val = res['final_model'].decision_function(point)[0]
                    
                    # Store absolute value of decision (influence magnitude)
                    importance_grid[y_idx, x_idx] = abs(decision_val)
            
            # Plot heatmap
            im = ax_imp[i].imshow(importance_grid, cmap='viridis', origin='lower', aspect='auto',
                                 extent=[np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)])
            
            # Convert standardized values to original scale for tick labels
            x_ticks = np.array([-2, -1, 0, 1, 2])
            y_ticks = np.array([-2, -1, 0, 1, 2])
            
            # Convert to original scale
            x_tick_labels = [f"{int(np.exp(res['scaler'].inverse_transform([[x, 0]])[0][0]) - 1):,}" for x in x_ticks]
            
            # Get date range from scaler
            min_date = model_df['release_date'].min()
            max_date = model_df['release_date'].max()
            date_range = (max_date - min_date).days
            mid_date = min_date + pd.Timedelta(days=date_range/2)
            
            # Create date tick labels
            y_tick_dates = [min_date + pd.Timedelta(days=int(y * date_range / 4 + date_range / 2)) for y in y_ticks]
            y_tick_labels = [d.strftime('%Y') for d in y_tick_dates]
            
            ax_imp[i].set_xticks(x_ticks)
            ax_imp[i].set_yticks(y_ticks)
            ax_imp[i].set_xticklabels(x_tick_labels)
            ax_imp[i].set_yticklabels(y_tick_labels)
            
            ax_imp[i].set_title(f'Feature Influence for {threshold}+ Recent Reviews')
            ax_imp[i].set_xlabel('Total Reviews')
            
            if i == 0:
                ax_imp[i].set_ylabel('Release Year')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_imp[i], label='Decision Influence')
        
        fig_imp.tight_layout()
        
        if save_path:
            fig_imp.savefig(f"{save_path}/feature_influence.png", dpi=300, bbox_inches='tight')
    
    # Return all results
    experiment_results = {
        'results': results,
        'figures': {
            'roc': fig_roc,
            'confusion_matrix': fig_cm,
            'decision_boundary': fig_boundary,
            'feature_importance': fig_imp if results else None
        }
    }
    
    return experiment_results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

def visualize_market_value_engagement(df, figsize=(16, 10), save_path=None):
    """
    Create a visualization of Steam games where:
    - Color represents market value (total reviews × price)
    - Size represents recent review brackets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 10))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set the style
    plt.style.use('ggplot')
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Calculate market value
    plot_df['market_value'] = plot_df['num_reviews_total'] * plot_df['price']
    
    # Log transform market value for better color mapping (add 1 to handle zeros)
    plot_df['log_market_value'] = np.log1p(plot_df['market_value'])
    
    # Create recent review brackets
    def recent_review_bracket(reviews):
        if reviews == 0:
            return 0  # No recent reviews
        elif reviews < 10:
            return 1  # 1-9 recent reviews
        elif reviews < 100:
            return 2  # 10-99 recent reviews
        elif reviews < 1000:
            return 3  # 100-999 recent reviews
        else:
            return 4  # 1000+ recent reviews
    
    plot_df['recent_review_bracket'] = plot_df['num_reviews_recent'].apply(recent_review_bracket)
    
    # Define size mapping for recent review brackets
    size_mapping = {
        0: 10,    # No recent reviews
        1: 25,    # 1-9 recent reviews
        2: 50,    # 10-99 recent reviews
        3: 100,   # 100-999 recent reviews
        4: 200    # 1000+ recent reviews
    }
    
    # Map bracket to size
    plot_df['point_size'] = plot_df['recent_review_bracket'].map(size_mapping)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=plot_df['log_market_value'].max())
    
    # Create scatter plot
    scatter = ax.scatter(
        plot_df['num_reviews_total'],
        plot_df['release_date'],
        s=plot_df['point_size'],
        c=plot_df['log_market_value'],
        cmap=cmap,
        alpha=0.7,
        edgecolor='none'
    )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax.set_ylabel('Release Date', fontsize=14)
    ax.set_title('Steam Games: Market Value and Review Activity', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log(Market Value) = Log(Total Reviews × Price + 1)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add total review count reference lines
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= plot_df['num_reviews_total'].max():
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.4)
            y_pos = ax.get_ylim()[0]  # Bottom of the plot
            ax.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Create a custom legend for point sizes
    bracket_labels = {
        0: "No recent reviews",
        1: "1-9 recent reviews",
        2: "10-99 recent reviews",
        3: "100-999 recent reviews",
        4: "1000+ recent reviews"
    }
    
    legend_elements = []
    for bracket, size in size_mapping.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=np.sqrt(size)/2,  # Adjust size for legend
                   label=bracket_labels[bracket])
        )
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper left', title="Recent Review Activity", fontsize=10)
    
    # Add statistics text
    stats_text = "Market Value & Activity Stats:\n"
    
    # Calculate percentages for each bracket
    for bracket, label in bracket_labels.items():
        count = len(plot_df[plot_df['recent_review_bracket'] == bracket])
        percentage = count / len(plot_df) * 100
        stats_text += f"{label}: {count:,} games ({percentage:.1f}%)\n"
    
    # Add highest market value game
    top_game = plot_df.loc[plot_df['market_value'].idxmax()]
    stats_text += f"\nHighest market value:\n{top_game['market_value']:,.0f} (Reviews × Price)"
    
    # Add text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Example usage:
# df = pd.read_csv('path/to/steam_games_data.csv')
# fig, ax = visualize_market_value_engagement(df, save_path='market_value_visualization.png')
# plt.show()import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

def visualize_market_value_engagement(df, figsize=(16, 10), save_path=None):
    """
    Create a visualization of Steam games where:
    - Color represents market value (total reviews × price)
    - Size represents recent review brackets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 10))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set the style
    plt.style.use('ggplot')
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Calculate market value
    plot_df['market_value'] = plot_df['num_reviews_total'] * plot_df['price']
    
    # Log transform market value for better color mapping (add 1 to handle zeros)
    plot_df['log_market_value'] = np.log1p(plot_df['market_value'])
    
    # Create recent review brackets
    def recent_review_bracket(reviews):
        if reviews == 0:
            return 0  # No recent reviews
        elif reviews < 10:
            return 1  # 1-9 recent reviews
        elif reviews < 100:
            return 2  # 10-99 recent reviews
        elif reviews < 1000:
            return 3  # 100-999 recent reviews
        else:
            return 4  # 1000+ recent reviews
    
    plot_df['recent_review_bracket'] = plot_df['num_reviews_recent'].apply(recent_review_bracket)
    
    # Define size mapping for recent review brackets
    size_mapping = {
        0: 10,    # No recent reviews
        1: 25,    # 1-9 recent reviews
        2: 50,    # 10-99 recent reviews
        3: 100,   # 100-999 recent reviews
        4: 200    # 1000+ recent reviews
    }
    
    # Map bracket to size
    plot_df['point_size'] = plot_df['recent_review_bracket'].map(size_mapping)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=plot_df['log_market_value'].max())
    
    # Create scatter plot
    scatter = ax.scatter(
        plot_df['num_reviews_total'],
        plot_df['release_date'],
        s=plot_df['point_size'],
        c=plot_df['log_market_value'],
        cmap=cmap,
        alpha=0.7,
        edgecolor='none'
    )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(0.5, plot_df['num_reviews_total'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax.set_xlabel('Total Reviews (log scale)', fontsize=14)
    ax.set_ylabel('Release Date', fontsize=14)
    ax.set_title('Steam Games: Market Value and Review Activity', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log(Market Value) = Log(Total Reviews × Price + 1)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add total review count reference lines
    review_thresholds = [10, 100, 1000, 10000, 100000]
    for threshold in review_thresholds:
        if threshold <= plot_df['num_reviews_total'].max():
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.4)
            y_pos = ax.get_ylim()[0]  # Bottom of the plot
            ax.text(threshold*1.1, y_pos, f"{threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Create a custom legend for point sizes
    bracket_labels = {
        0: "No recent reviews",
        1: "1-9 recent reviews",
        2: "10-99 recent reviews",
        3: "100-999 recent reviews",
        4: "1000+ recent reviews"
    }
    
    legend_elements = []
    for bracket, size in size_mapping.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=np.sqrt(size)/2,  # Adjust size for legend
                   label=bracket_labels[bracket])
        )
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper left', title="Recent Review Activity", fontsize=10)
    
    # Add statistics text
    stats_text = "Market Value & Activity Stats:\n"
    
    # Calculate percentages for each bracket
    for bracket, label in bracket_labels.items():
        count = len(plot_df[plot_df['recent_review_bracket'] == bracket])
        percentage = count / len(plot_df) * 100
        stats_text += f"{label}: {count:,} games ({percentage:.1f}%)\n"
    
    # Add highest market value game
    top_game = plot_df.loc[plot_df['market_value'].idxmax()]
    stats_text += f"\nHighest market value:\n{top_game['market_value']:,.0f} (Reviews × Price)"
    
    # Add text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_recent_activity_value(df, figsize=(16, 10), save_path=None):
    """
    Create a visualization of Steam games where:
    - X-axis is recent reviews * price (recent activity value)
    - Y-axis is release date
    - Color represents total reviews
    - Size represents recent review brackets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_total
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 10))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set the style
    plt.style.use('ggplot')
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_total'] = plot_df['num_reviews_total'].fillna(0)
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Calculate recent activity value
    plot_df['recent_activity_value'] = plot_df['num_reviews_recent'] * plot_df['price']
    
    # Log transform total reviews for better color mapping
    plot_df['log_total_reviews'] = np.log1p(plot_df['num_reviews_total'])
    
    # Create recent review brackets for sizing points
    def recent_review_bracket(reviews):
        if reviews == 0:
            return 0  # No recent reviews
        elif reviews < 10:
            return 1  # 1-9 recent reviews
        elif reviews < 100:
            return 2  # 10-99 recent reviews
        elif reviews < 1000:
            return 3  # 100-999 recent reviews
        else:
            return 4  # 1000+ recent reviews
    
    plot_df['recent_review_bracket'] = plot_df['num_reviews_recent'].apply(recent_review_bracket)
    
    # Define size mapping for recent review brackets
    size_mapping = {
        0: 10,    # No recent reviews
        1: 25,    # 1-9 recent reviews
        2: 50,    # 10-99 recent reviews
        3: 100,   # 100-999 recent reviews
        4: 200    # 1000+ recent reviews
    }
    
    # Map bracket to size
    plot_df['point_size'] = plot_df['recent_review_bracket'].map(size_mapping)
    
    # Filter to games with at least some recent activity value
    # Add a small value to include games with zero but display properly on log scale
    plot_df['plot_value'] = plot_df['recent_activity_value'] + 0.1
    active_df = plot_df[plot_df['num_reviews_recent'] > 0].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap for total reviews
    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=plot_df['log_total_reviews'].max())
    
    # Create scatter plot
    scatter = ax.scatter(
        active_df['plot_value'],
        active_df['release_date'],
        s=active_df['point_size'],
        c=active_df['log_total_reviews'],
        cmap=cmap,
        alpha=0.7,
        edgecolor='none'
    )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    min_value = 0.1  # slightly above 0 for log scale
    ax.set_xlim(min_value, active_df['plot_value'].max() * 1.1)
    
    # Format the y-axis as dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add labels and title
    ax.set_xlabel('Recent Activity Value (Recent Reviews × Price, log scale)', fontsize=14)
    ax.set_ylabel('Release Date', fontsize=14)
    ax.set_title('Steam Games: Recent Activity Value by Release Date', fontsize=16)
    
    # Add colorbar for total reviews
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log(Total Reviews + 1)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add reference lines for recent activity value
    value_thresholds = [1, 10, 100, 1000, 10000, 100000]
    for threshold in value_thresholds:
        if threshold <= active_df['recent_activity_value'].max():
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.4)
            y_pos = ax.get_ylim()[0]  # Bottom of the plot
            ax.text(threshold*1.1, y_pos, f"${threshold:,}", 
                    fontsize=9, rotation=90, verticalalignment='bottom')
    
    # Create a custom legend for point sizes
    bracket_labels = {
        0: "No recent reviews",
        1: "1-9 recent reviews",
        2: "10-99 recent reviews",
        3: "100-999 recent reviews",
        4: "1000+ recent reviews"
    }
    
    legend_elements = []
    for bracket, size in size_mapping.items():
        if bracket > 0:  # Skip the "No recent reviews" bracket since we filtered them out
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=np.sqrt(size)/2,  # Adjust size for legend
                       label=bracket_labels[bracket])
            )
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper right', title="Recent Review Activity", fontsize=10)
    
    # Add statistics text
    stats_text = "Activity Value Stats:\n"
    
    # Calculate percentages for each bracket (of active games)
    active_count = len(active_df)
    inactive_count = len(plot_df) - active_count
    inactive_pct = inactive_count / len(plot_df) * 100
    stats_text += f"Games with no recent reviews: {inactive_count:,} ({inactive_pct:.1f}%)\n\n"
    
    for bracket in range(1, 5):  # Skip bracket 0 (no reviews)
        count = len(active_df[active_df['recent_review_bracket'] == bracket])
        percentage = count / len(plot_df) * 100
        stats_text += f"{bracket_labels[bracket]}: {count:,} games ({percentage:.1f}%)\n"
    
    # Add highest value game
    top_game = active_df.loc[active_df['recent_activity_value'].idxmax()]
    stats_text += f"\nHighest activity value:\n${top_game['recent_activity_value']:,.0f} (Recent Reviews × Price)"
    
    # Add text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

def visualize_market_volume_kde(df, figsize=(16, 10), save_path=None):
    """
    Create a KDE visualization of market volume (recent reviews × price) over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data with required columns:
        - release_date
        - num_reviews_recent
        - price
    figsize : tuple, optional
        Size of the figure (default: (16, 10))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Set the style
    plt.style.use('ggplot')
    
    # Create a copy and ensure data types
    plot_df = df.copy()
    
    # Ensure release_date is datetime
    plot_df['release_date'] = pd.to_datetime(plot_df['release_date'], errors='coerce')
    
    # Fill missing values
    plot_df['num_reviews_recent'] = plot_df['num_reviews_recent'].fillna(0)
    plot_df['price'] = plot_df['price'].fillna(0)
    
    # Calculate market volume
    plot_df['market_volume'] = plot_df['num_reviews_recent'] * plot_df['price']
    
    # Filter to games with some market volume (recent reviews > 0 and price > 0)
    active_df = plot_df[(plot_df['num_reviews_recent'] > 0) & (plot_df['price'] > 0)].copy()
    
    # Create numeric version of release date for KDE
    active_df['release_date_numeric'] = active_df['release_date'].apply(lambda x: x.toordinal())
    
    # Log transform market volume for better visualization
    active_df['log_market_volume'] = np.log1p(active_df['market_volume'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the market volume and release date data
    x = active_df['release_date_numeric'].values
    y = active_df['log_market_volume'].values
    
    # Create a 2D KDE
    # First, convert to numpy arrays
    xy = np.vstack([x, y])
    
    try:
        # Try to estimate the kernel density
        kernel = stats.gaussian_kde(xy)
        
        # Create a grid for evaluation
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # Evaluate the KDE at the grid positions
        Z = kernel(positions).reshape(X.shape)
        
        # Convert the x-grid back to dates
        x_dates = [pd.Timestamp.fromordinal(int(ordinal)) for ordinal in x_grid]
        
        # Convert the y-grid back from log scale
        y_values = np.exp(y_grid) - 1
        
        # Create a custom colormap from blue to red
        colors = [(0, 'royalblue'), (0.5, 'purple'), (1, 'crimson')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Plot the KDE as a filled contour map
        contour = ax.contourf(x_dates, y_grid, Z, levels=50, cmap=custom_cmap, alpha=0.7)
        
        # Add a color bar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Density', fontsize=12)
        
        # Add a scatter plot with reduced alpha to show individual games
        scatter = ax.scatter(
            active_df['release_date'],
            active_df['log_market_volume'],
            s=10,
            c='white',
            alpha=0.1
        )
        
        # Format the x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        # Create custom y-ticks for log-transformed market volume
        y_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_tick_labels = [f"${int(np.exp(y) - 1):,}" for y in y_ticks]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        # Add labels and title
        ax.set_xlabel('Release Date', fontsize=14)
        ax.set_ylabel('Market Volume (Recent Reviews × Price)', fontsize=14)
        ax.set_title('KDE of Steam Game Market Volume Over Time', fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = "Market Volume Statistics:\n"
        stats_text += f"Active games (with market volume): {len(active_df):,}\n"
        stats_text += f"Inactive games: {len(plot_df) - len(active_df):,}\n\n"
        
        # Add quantile information
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.99]
        for q in quantiles:
            quantile_value = np.quantile(active_df['market_volume'], q)
            stats_text += f"{int(q*100)}th percentile: ${quantile_value:.2f}\n"
        
        # Add top market volume games
        top_game = active_df.loc[active_df['market_volume'].idxmax()]
        stats_text += f"\nHighest market volume: ${top_game['market_volume']:,.2f}"
        
        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Add annotations for high density regions
        # Find the maximum density point
        max_idx = np.argmax(Z)
        max_x_idx, max_y_idx = np.unravel_index(max_idx, Z.shape)
        max_date = x_dates[max_x_idx]
        max_log_volume = y_grid[max_y_idx]
        
        ax.annotate('Highest Density Region',
                   xy=(max_date, max_log_volume),
                   xytext=(max_date - pd.Timedelta(days=365), max_log_volume + 1),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    except Exception as e:
        # If KDE fails, fallback to a simple scatter plot
        print(f"KDE calculation failed: {e}")
        print("Falling back to scatter plot")
        
        scatter = ax.scatter(
            active_df['release_date'],
            active_df['log_market_volume'],
            s=10,
            c=active_df['market_volume'],
            cmap='viridis',
            alpha=0.7
        )
        
        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Market Volume', fontsize=12)
        
        # Format the x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        # Create custom y-ticks for log-transformed market volume
        y_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_tick_labels = [f"${int(np.exp(y) - 1):,}" for y in y_ticks]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        # Add labels and title
        ax.set_xlabel('Release Date', fontsize=14)
        ax.set_ylabel('Market Volume (Recent Reviews × Price, log scale)', fontsize=14)
        ax.set_title('Steam Game Market Volume Over Time', fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

# Example usage:
# df = pd.read_csv('path/to/steam_games_data.csv')
# fig, ax = visualize_market_volume_kde(df, save_path='market_volume_kde.png')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

def analyze_steam_market_competitors(df, recent_review_threshold=10, max_clusters=5, return_figures=True):
    """
    Analyze Steam games dataset to identify market competitors and success factors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data
    recent_review_threshold : int, optional
        Threshold of recent reviews to consider a game as a market competitor (default: 10)
    max_clusters : int, optional
        Number of clusters for market segmentation analysis (default: 5)
    return_figures : bool, optional
        Whether to return matplotlib figures (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and optionally figures
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # ---- Data Preparation ----
    # Convert release_date to datetime
    if 'release_date' in data.columns:
        data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
        data['year'] = data['release_date'].dt.year
        data['month'] = data['release_date'].dt.month
        data['release_day_of_week'] = data['release_date'].dt.dayofweek
        data['release_quarter'] = (data['release_date'].dt.month - 1) // 3 + 1
        data['days_since_release'] = (datetime.now() - data['release_date']).dt.days
    
    # Identify self-published games
    def is_self_published(row):
        """Check if a game is self-published by comparing developers and publishers."""
        developers = row.get('developers', [])
        publishers = row.get('publishers', [])
        
        # Handle missing values
        if pd.isna(developers) or pd.isna(publishers):
            return False
        
        # Convert string representations of lists to actual lists
        for field in ['developers', 'publishers']:
            value = row.get(field, '')
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        row[field] = eval(value)
                    except:
                        row[field] = [value]
                else:
                    row[field] = [value]
        
        # Check for overlap
        if isinstance(row.get('developers', []), list) and isinstance(row.get('publishers', []), list):
            return any(dev in row['publishers'] for dev in row['developers'])
        
        return False
    
    data['is_self_published'] = data.apply(is_self_published, axis=1)
    
    # Find review columns
    recent_review_col = 'num_reviews_recent' if 'num_reviews_recent' in data.columns else None
    total_review_col = 'num_reviews_total' if 'num_reviews_total' in data.columns else None
    
    if not recent_review_col:
        # Try to find an alternative column
        review_cols = [col for col in data.columns if 'review' in col.lower() and 'recent' in col.lower()]
        if review_cols:
            recent_review_col = review_cols[0]
        else:
            raise ValueError("No recent review column found. Cannot identify market competitors.")
    
    # ---- Market Competitor Identification ----
    # Define competitors and create target variable
    data['is_competitor'] = data[recent_review_col] > recent_review_threshold
    competitors = data[data['is_competitor']].copy()
    non_competitors = data[~data['is_competitor']].copy()
    
    # ---- Feature Engineering ----
    # Calculate positive review percentage if available
    if 'positive' in data.columns and 'negative' in data.columns:
        total_reviews = data['positive'] + data['negative']
        data['positive_pct'] = data['positive'] / (total_reviews + 1) * 100  # +1 to avoid division by zero
    
    # Extract metadata counts
    for field in ['developers', 'publishers', 'screenshots', 'movies']:
        if field in data.columns:
            def count_entries(entry_str):
                try:
                    if isinstance(entry_str, str):
                        if entry_str.startswith('[') and entry_str.endswith(']'):
                            entries = eval(entry_str)
                            return len(entries)
                        return 1
                    elif isinstance(entry_str, list):
                        return len(entry_str)
                except:
                    pass
                return 0
            
            data[f'{field}_count'] = data[field].apply(count_entries)
    
    # Calculate developer/publisher ratio if both fields exist
    if 'developers_count' in data.columns and 'publishers_count' in data.columns:
        data['dev_pub_ratio'] = data['developers_count'] / (data['publishers_count'] + 0.1)
    
    # Extract categorical features (genres, categories, tags)
    categorical_features = []
    for field in ['genres', 'categories', 'tags']:
        if field in data.columns:
            def extract_items(item_str):
                try:
                    if isinstance(item_str, str):
                        if item_str.startswith('[') and item_str.endswith(']'):
                            return eval(item_str)
                        return [item_str]
                    elif isinstance(item_str, list):
                        return item_str
                except:
                    pass
                return []
            
            data[f'extracted_{field}'] = data[field].apply(extract_items)
            categorical_features.append(field)
    
    # ---- Basic Statistics ----
    stats = {
        'total_games': len(data),
        'competitors': len(competitors),
        'competitor_pct': len(competitors) / len(data) * 100,
        'avg_price': {
            'competitors': competitors['price'].mean() if 'price' in competitors else None,
            'non_competitors': non_competitors['price'].mean() if 'price' in non_competitors else None
        },
        'self_published': {
            'competitors': competitors['is_self_published'].mean() * 100 if 'is_self_published' in competitors else None,
            'non_competitors': non_competitors['is_self_published'].mean() * 100 if 'is_self_published' in non_competitors else None
        }
    }
    
    # ---- Correlation Analysis ----
    # Valid numeric features for correlation analysis
    exclude_features = [
        'num_reviews_total', 'recommendations', 'average_playtime_forever',
        'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks'
    ]
    
    numeric_cols = data.select_dtypes(include=['number']).columns
    valid_features = [col for col in numeric_cols if col not in exclude_features and col != 'is_competitor']
    
    # Calculate correlation with competitor status
    correlations = []
    for feature in valid_features:
        if data[feature].nunique() > 1:
            correlation = data[[feature, 'is_competitor']].corr().iloc[0, 1]
            correlations.append({
                'Feature': feature,
                'Correlation': correlation
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    
    # ---- Success Rate Analysis ----
    success_rates = {}
    
    # By genre
    if 'extracted_genres' in data.columns:
        genre_data = []
        all_genres = []
        
        for genres in data['extracted_genres']:
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(15).index.tolist()
        
        for genre in top_genres:
            genre_games = data[data['genres'].apply(lambda x: isinstance(x, str) and genre in x)]
            genre_competitors = competitors[competitors['genres'].apply(lambda x: isinstance(x, str) and genre in x)]
            
            if len(genre_games) > 10:  # Only consider genres with enough data
                success_rate = len(genre_competitors) / len(genre_games) * 100
                genre_data.append({
                    'Genre': genre,
                    'Total': len(genre_games),
                    'Competitors': len(genre_competitors),
                    'Success_Rate': success_rate
                })
        
        success_rates['genres'] = pd.DataFrame(genre_data).sort_values('Success_Rate', ascending=False)
    
    # By price band
    price_bands = [
        (0, 0, "Free"),
        (0.01, 4.99, "$0.01-$4.99"),
        (5, 9.99, "$5.00-$9.99"),
        (10, 14.99, "$10.00-$14.99"),
        (15, 19.99, "$15.00-$19.99"),
        (20, 29.99, "$20.00-$29.99"),
        (30, 39.99, "$30.00-$39.99"),
        (40, 49.99, "$40.00-$49.99"),
        (50, 59.99, "$50.00-$59.99"),
        (60, float('inf'), "$60.00+")
    ]
    
    price_data = []
    for lower, upper, label in price_bands:
        if lower == 0 and upper == 0:
            band_games = data[data['price'] == 0]
            band_competitors = competitors[competitors['price'] == 0]
        else:
            band_games = data[(data['price'] > lower) & (data['price'] <= upper)]
            band_competitors = competitors[(competitors['price'] > lower) & (competitors['price'] <= upper)]
        
        if len(band_games) > 0:
            success_rate = len(band_competitors) / len(band_games) * 100
            price_data.append({
                'Price_Band': label,
                'Total': len(band_games),
                'Competitors': len(band_competitors),
                'Success_Rate': success_rate,
                'Avg_Positive': band_competitors['positive_pct'].mean() if 'positive_pct' in band_competitors.columns and len(band_competitors) > 0 else None
            })
    
    success_rates['price_bands'] = pd.DataFrame(price_data)
    
    # By release timing
    if 'release_date' in data.columns:
        # Day of week
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = []
        
        for day_num in range(7):
            day_games = data[data['release_date'].dt.dayofweek == day_num]
            day_competitors = competitors[competitors['release_date'].dt.dayofweek == day_num]
            
            if len(day_games) > 0:
                success_rate = len(day_competitors) / len(day_games) * 100
                day_data.append({
                    'Day': day_names[day_num],
                    'Total': len(day_games),
                    'Competitors': len(day_competitors),
                    'Success_Rate': success_rate
                })
        
        success_rates['days'] = pd.DataFrame(day_data)
        
        # Month
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_data = []
        
        for month_num in range(1, 13):
            month_games = data[data['release_date'].dt.month == month_num]
            month_competitors = competitors[competitors['release_date'].dt.month == month_num]
            
            if len(month_games) > 0:
                success_rate = len(month_competitors) / len(month_games) * 100
                month_data.append({
                    'Month': month_names[month_num-1],
                    'Total': len(month_games),
                    'Competitors': len(month_competitors),
                    'Success_Rate': success_rate
                })
        
        success_rates['months'] = pd.DataFrame(month_data)
    
    # ---- Cluster Analysis ----
    cluster_results = None
    figures = {}
    
    # Get features for clustering
    exclude_cols = [recent_review_col, total_review_col, 'is_competitor']
    cluster_features = [col for col in valid_features if col not in exclude_cols and col in competitors.columns]
    
    if len(cluster_features) >= 2 and len(competitors) >= max_clusters * 5:  # Ensure enough data for clustering
        # Handle missing values and prepare data
        cluster_df = competitors[cluster_features].fillna(0)
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=max_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        competitors['cluster'] = clusters
        
        # Analyze clusters
        cluster_stats = competitors.groupby('cluster').agg({
            recent_review_col: ['mean', 'median'],
            'price': ['mean', 'median'] if 'price' in competitors else ['count'],
            'is_self_published': 'mean' if 'is_self_published' in competitors else 'count'
        }).reset_index()
        
        # Format column names
        cluster_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cluster_stats.columns]
        cluster_stats['count'] = competitors.groupby('cluster').size().values
        
        cluster_results = {
            'stats': cluster_stats,
            'feature_importance': pd.DataFrame({
                'feature': cluster_features,
                'importance': np.abs(kmeans.cluster_centers_).mean(axis=0)
            }).sort_values('importance', ascending=False)
        }
        
        # Top genres by cluster if available
        if 'extracted_genres' in competitors.columns:
            cluster_genres = {}
            for cluster_id in range(max_clusters):
                cluster_games = competitors[competitors['cluster'] == cluster_id]
                
                # Get all genres in this cluster
                all_cluster_genres = []
                for genres in cluster_games['extracted_genres']:
                    all_cluster_genres.extend(genres)
                
                if all_cluster_genres:
                    # Count occurrences
                    genre_counts = pd.Series(all_cluster_genres).value_counts().head(5)
                    cluster_genres[cluster_id] = genre_counts.to_dict()
            
            cluster_results['top_genres'] = cluster_genres
        
        # Create visualization
        if return_figures:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, 
                              cmap='viridis', alpha=0.5, s=50)
            plt.colorbar(scatter, label='Cluster')
            ax.set_title('Market Competitor Clusters (PCA Reduced)', fontsize=16)
            ax.set_xlabel('Principal Component 1', fontsize=12)
            ax.set_ylabel('Principal Component 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            figures['clusters'] = fig
    
    # ---- Create Recommendations ----
    recommendations = {}
    
    # Optimal price points
    if 'price_bands' in success_rates:
        top_price = success_rates['price_bands'].sort_values('Success_Rate', ascending=False).head(3)
        recommendations['price_points'] = top_price
    
    # Most successful genres
    if 'genres' in success_rates:
        top_genres = success_rates['genres'].head(5)
        recommendations['genres'] = top_genres
    
    # Optimal release timing
    timing_recs = {}
    if 'days' in success_rates:
        timing_recs['days'] = success_rates['days'].sort_values('Success_Rate', ascending=False).head(3)
    if 'months' in success_rates:
        timing_recs['months'] = success_rates['months'].sort_values('Success_Rate', ascending=False).head(3)
    
    if timing_recs:
        recommendations['timing'] = timing_recs
    
    # Key features for success
    recommendations['key_features'] = corr_df[corr_df['Correlation'] > 0].head(5)
    
    # Create result dictionary
    results = {
        'stats': stats,
        'correlations': corr_df,
        'success_rates': success_rates,
        'recommendations': recommendations
    }
    
    if cluster_results:
        results['cluster_analysis'] = cluster_results
    
    if return_figures:
        results['figures'] = figures
    
    return results

# Example usage:
# import pandas as pd
# df = pd.read_csv('steam_games_data.csv')
# analysis = analyze_steam_market_competitors(df)
# 
# # View results
# print(f"Found {analysis['stats']['competitors']} market competitors ({analysis['stats']['competitor_pct']:.2f}%)")
# print("\nTop features correlated with success:")
# print(analysis['correlations'].head(5))
# 
# # Best price points
# print("\nMost successful price points:")
# print(analysis['recommendations']['price_points'][['Price_Band', 'Success_Rate']].head(3))
# 
# # Show cluster visualization
# if 'figures' in analysis and 'clusters' in analysis['figures']:
#     analysis['figures']['clusters'].show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import time

def predict_market_competitors(df, recent_review_threshold=10, n_splits=5, return_figures=True, 
                              n_estimators=100, max_depth=None, svc_c=1.0, svc_gamma='scale'):
    """
    Run lightweight cross-validation for Random Forest and SVC models to predict market competitors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data
    recent_review_threshold : int, optional
        Threshold of recent reviews to consider a game as a market competitor (default: 10)
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    return_figures : bool, optional
        Whether to return matplotlib figures (default: True)
    n_estimators : int, optional
        Number of trees in the Random Forest (default: 100)
    max_depth : int or None, optional
        Maximum depth of trees in Random Forest (default: None)
    svc_c : float, optional
        Regularization parameter for SVC (default: 1.0)
    svc_gamma : str or float, optional
        Kernel coefficient for SVC (default: 'scale')
        
    Returns:
    --------
    dict
        Dictionary containing model performance metrics and optionally figures
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Find recent review column
    recent_review_col = 'num_reviews_recent' if 'num_reviews_recent' in data.columns else None
    total_review_col = 'num_reviews_total' if 'num_reviews_total' in data.columns else None
    
    if not recent_review_col:
        # Try to find an alternative column
        review_cols = [col for col in data.columns if 'review' in col.lower() and 'recent' in col.lower()]
        if review_cols:
            recent_review_col = review_cols[0]
        else:
            raise ValueError("No recent review column found. Cannot identify market competitors.")
    
    # Create target variable
    data['is_competitor'] = (data[recent_review_col] > recent_review_threshold).astype(int)
    
    # ---- Feature Engineering ----
    
    # 1. Basic features - ensure they're all numeric
    exclude_features = [
        recent_review_col, total_review_col, 'num_reviews_total', 'recommendations', 'average_playtime_forever',
        'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks'
    ]
    
    # Basic preparatory feature engineering
    if 'release_date' in data.columns:
        data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
        data['days_since_release'] = (pd.Timestamp.now() - data['release_date']).dt.days
        data['release_day_of_week'] = data['release_date'].dt.dayofweek
        data['release_month'] = data['release_date'].dt.month
    
    # Create binary flags for platforms
    for platform in ['windows', 'mac', 'linux']:
        if platform in data.columns:
            data[platform] = data[platform].astype(str).str.lower().map({'true': 1, 'false': 0})
            data[platform] = pd.to_numeric(data[platform], errors='coerce').fillna(0).astype(int)
    
    # Process metacritic score (if available)
    if 'metacritic_score' in data.columns:
        data['has_metacritic'] = (~data['metacritic_score'].isna()).astype(int)
    
    # Get all numeric features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Filter features
    valid_features = [col for col in numeric_cols if col not in exclude_features and col != 'is_competitor']
    
    # Count missing values and handle them
    missing_counts = data[valid_features].isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    # Fill missing values
    for feature in features_with_missing:
        data[feature] = data[feature].fillna(data[feature].median())
    
    # Prepare X and y
    X = data[valid_features]
    y = data['is_competitor']
    
    # Initialize models
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    svc_model = SVC(
        C=svc_c,
        gamma=svc_gamma,
        probability=True,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    # Initialize results dictionary
    results = {
        'feature_count': len(valid_features),
        'class_distribution': {
            'competitors': y.sum(),
            'non_competitors': len(y) - y.sum(),
            'imbalance_ratio': (len(y) - y.sum()) / y.sum() if y.sum() > 0 else float('inf')
        },
        'models': {}
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Pre-scale data for SVC
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # List to store figures
    figures = {}
    
    # ---- Run Cross-Validation for Random Forest ----
    print(f"Running {n_splits}-fold cross-validation for Random Forest...")
    start_time = time.time()
    
    # Get cross-validated predictions
    y_pred_rf = cross_val_predict(rf_model, X, y, cv=cv)
    y_true = y  # For clarity
    
    # Calculate performance metrics
    rf_metrics = {
        'accuracy': accuracy_score(y_true, y_pred_rf),
        'precision': precision_score(y_true, y_pred_rf),
        'recall': recall_score(y_true, y_pred_rf),
        'f1': f1_score(y_true, y_pred_rf),
        'training_time': time.time() - start_time
    }
    
    # Create confusion matrix
    rf_cm = confusion_matrix(y_true, y_pred_rf)
    rf_cm_normalized = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]
    
    # Train a final model on all data for feature importance
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    final_rf.fit(X, y)
    
    # Get feature importances
    importances = final_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a DataFrame with feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns[indices],
        'Importance': importances[indices]
    })
    
    # Store results
    results['models']['random_forest'] = {
        'metrics': rf_metrics,
        'confusion_matrix': rf_cm.tolist(),
        'feature_importance': feature_importance_df.head(15).to_dict(orient='records'),
        'all_feature_importance': feature_importance_df.to_dict(orient='records')
    }
    
    # Create visualization of RF feature importance
    if return_figures:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(20, len(feature_importance_df))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance_df.head(top_n),
            palette='viridis',
            ax=ax
        )
        ax.set_title(f'Top {top_n} Features for Predicting Market Competitors (Random Forest)', fontsize=16)
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        figures['rf_importance'] = fig
    
    # ---- Run Cross-Validation for SVC ----
    print(f"Running {n_splits}-fold cross-validation for SVC...")
    start_time = time.time()
    
    # Get cross-validated predictions
    y_pred_svc = cross_val_predict(svc_model, X_scaled_df, y, cv=cv)
    
    # Calculate performance metrics
    svc_metrics = {
        'accuracy': accuracy_score(y_true, y_pred_svc),
        'precision': precision_score(y_true, y_pred_svc),
        'recall': recall_score(y_true, y_pred_svc),
        'f1': f1_score(y_true, y_pred_svc),
        'training_time': time.time() - start_time
    }
    
    # Create confusion matrix
    svc_cm = confusion_matrix(y_true, y_pred_svc)
    svc_cm_normalized = svc_cm.astype('float') / svc_cm.sum(axis=1)[:, np.newaxis]
    
    # Train a final model on all data for feature importance via permutation importance
    final_svc = SVC(
        C=svc_c,
        gamma=svc_gamma,
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    final_svc.fit(X_scaled_df, y)
    
    # Use permutation importance for SVC (can be computationally expensive, so limit to top 10 features)
    try:
        # Use a subset of data if the dataset is very large
        if len(X) > 10000:
            sample_size = min(5000, len(X) // 2)
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X_scaled_df.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X_scaled_df
            y_sample = y
        
        # Calculate permutation importance (limit to top features from RF for efficiency)
        top_rf_features = feature_importance_df['Feature'].head(15).tolist()
        X_sample_top = X_sample[top_rf_features]
        
        perm_importance = permutation_importance(
            final_svc, X_sample_top, y_sample,
            n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Create a DataFrame for SVC feature importance
        svc_importance_df = pd.DataFrame({
            'Feature': top_rf_features,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        results['models']['svc']['feature_importance'] = svc_importance_df.to_dict(orient='records')
        
        # Visualization for SVC feature importance
        if return_figures:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=svc_importance_df,
                palette='crest',
                ax=ax
            )
            ax.set_title('Feature Importance for Predicting Market Competitors (SVC)', fontsize=16)
            ax.set_xlabel('Permutation Importance', fontsize=14)
            ax.set_ylabel('Feature', fontsize=14)
            plt.tight_layout()
            figures['svc_importance'] = fig
            
    except Exception as e:
        print(f"Skipping SVC permutation importance due to error: {e}")
        results['models']['svc']['feature_importance'] = "Skipped due to computational constraints"
    
    # Store results
    results['models']['svc'] = {
        'metrics': svc_metrics,
        'confusion_matrix': svc_cm.tolist()
    }
    
    # ---- Create performance comparison visualization ----
    if return_figures:
        # Compare model metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Random Forest': [rf_metrics['accuracy'], rf_metrics['precision'], 
                             rf_metrics['recall'], rf_metrics['f1']],
            'SVC': [svc_metrics['accuracy'], svc_metrics['precision'], 
                   svc_metrics['recall'], svc_metrics['f1']]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.set_index('Metric').plot(kind='bar', ax=ax)
        ax.set_title('Performance Comparison: Random Forest vs SVC', fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score', fontsize=14)
        ax.legend(fontsize=12)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
            
        plt.tight_layout()
        figures['model_comparison'] = fig
        
        # Create confusion matrix visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Random Forest confusion matrix
        sns.heatmap(
            rf_cm_normalized, annot=rf_cm, fmt='d',
            cmap='Blues', cbar=False, ax=axes[0]
        )
        axes[0].set_title('Random Forest Confusion Matrix', fontsize=14)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xticklabels(['Non-Competitor', 'Competitor'])
        axes[0].set_yticklabels(['Non-Competitor', 'Competitor'])
        
        # SVC confusion matrix
        sns.heatmap(
            svc_cm_normalized, annot=svc_cm, fmt='d',
            cmap='Blues', cbar=False, ax=axes[1]
        )
        axes[1].set_title('SVC Confusion Matrix', fontsize=14)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xticklabels(['Non-Competitor', 'Competitor'])
        axes[1].set_yticklabels(['Non-Competitor', 'Competitor'])
        
        plt.tight_layout()
        figures['confusion_matrices'] = fig
    
    # Add figures to results if requested
    if return_figures:
        results['figures'] = figures
    
    # ---- Summary Report ----
    # Create descriptive summary
    summary = (
        f"Model Comparison Summary:\n"
        f"------------------------\n"
        f"Data: {len(X)} games, {results['class_distribution']['competitors']} competitors "
        f"({results['class_distribution']['competitors']/len(X)*100:.1f}%)\n"
        f"Features: {len(valid_features)} features used for prediction\n\n"
        f"Random Forest Performance:\n"
        f"  Accuracy: {rf_metrics['accuracy']:.3f}\n"
        f"  Precision: {rf_metrics['precision']:.3f}\n"
        f"  Recall: {rf_metrics['recall']:.3f}\n"
        f"  F1 Score: {rf_metrics['f1']:.3f}\n\n"
        f"SVC Performance:\n"
        f"  Accuracy: {svc_metrics['accuracy']:.3f}\n"
        f"  Precision: {svc_metrics['precision']:.3f}\n"
        f"  Recall: {svc_metrics['recall']:.3f}\n"
        f"  F1 Score: {svc_metrics['f1']:.3f}\n\n"
        f"Top 5 Predictive Features:\n"
    )
    
    for i, row in feature_importance_df.head(5).iterrows():
        summary += f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}\n"
    
    results['summary'] = summary
    print(summary)
    
    return results

# Example usage:
# import pandas as pd
# df = pd.read_csv('steam_games_data.csv')
# model_results = predict_market_competitors(df)
# 
# # Access performance metrics
# rf_metrics = model_results['models']['random_forest']['metrics']
# svc_metrics = model_results['models']['svc']['metrics']
# 
# # Get top predictive features
# top_features = model_results['models']['random_forest']['feature_importance']
# 
# # Display figures
# if 'figures' in model_results:
#     model_results['figures']['model_comparison'].show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

def analyze_steam_competitors_rf(df, recent_review_threshold=10, n_splits=5, n_estimators=100, 
                                max_depth=None, prob_threshold=0.6):
    """
    Run Random Forest to predict market competitors and analyze misclassifications.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data
    recent_review_threshold : int, optional
        Threshold of recent reviews to consider a game as a market competitor (default: 10)
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    n_estimators : int, optional
        Number of trees in the Random Forest (default: 100)
    max_depth : int or None, optional
        Maximum depth of trees in Random Forest (default: None)
    prob_threshold : float, optional
        Probability threshold for low-confidence analysis (default: 0.6)
        
    Returns:
    --------
    dict
        Dictionary containing model results and analysis of misclassifications
    """
    print(f"Running Random Forest analysis with {n_splits}-fold cross-validation...")
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Find recent review column
    recent_review_col = 'num_reviews_recent' if 'num_reviews_recent' in data.columns else None
    total_review_col = 'num_reviews_total' if 'num_reviews_total' in data.columns else None
    
    if not recent_review_col:
        # Try to find an alternative column
        review_cols = [col for col in data.columns if 'review' in col.lower() and 'recent' in col.lower()]
        if review_cols:
            recent_review_col = review_cols[0]
        else:
            raise ValueError("No recent review column found. Cannot identify market competitors.")
    
    # Create target variable
    data['is_competitor'] = (data[recent_review_col] > recent_review_threshold).astype(int)
    
    # Basic preparatory feature engineering
    if 'release_date' in data.columns:
        data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
        data['days_since_release'] = (pd.Timestamp.now() - data['release_date']).dt.days
        data['release_year'] = data['release_date'].dt.year
        data['release_month'] = data['release_date'].dt.month
        data['release_day_of_week'] = data['release_date'].dt.dayofweek
    
    # Create binary flags for platforms
    for platform in ['windows', 'mac', 'linux']:
        if platform in data.columns:
            if not pd.api.types.is_numeric_dtype(data[platform]):
                data[platform] = data[platform].astype(str).str.lower().map({'true': 1, 'false': 0})
                data[platform] = pd.to_numeric(data[platform], errors='coerce').fillna(0).astype(int)
    
    # Process metacritic score
    if 'metacritic_score' in data.columns:
        data['has_metacritic'] = (~data['metacritic_score'].isna()).astype(int)
    
    # Features to exclude
    exclude_features = [
        recent_review_col, total_review_col, 'num_reviews_total', 'recommendations', 'average_playtime_forever',
        'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks', 'is_competitor'
    ]
    
    # Get all potential numeric features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    valid_features = [col for col in numeric_cols if col not in exclude_features]
    
    # Fill missing values with median
    for feature in valid_features:
        if data[feature].isnull().sum() > 0:
            data[feature] = data[feature].fillna(data[feature].median())
    
    # Prepare X and y
    X = data[valid_features]
    y = data['is_competitor']
    
    # Initialize Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get cross-validated predictions with probabilities
    y_proba = cross_val_predict(rf_model, X, y, cv=cv, method='predict_proba')
    y_pred = (y_proba[:, 1] >= 0.5).astype(int)
    
    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Train a final model for feature importance
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    final_rf.fit(X, y)
    
    # Get feature importances
    importances = final_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importance = pd.DataFrame({
        'Feature': X.columns[indices],
        'Importance': importances[indices]
    })
    
    # ===== ANALYSIS OF MISCLASSIFICATIONS =====
    
    # Add predictions to the data
    data['predicted_probability'] = y_proba[:, 1]
    data['predicted_class'] = y_pred
    
    # 1. False Positives: Predicted as competitor but actually not
    false_positives = data[(data['predicted_class'] == 1) & (data['is_competitor'] == 0)].copy()
    
    # 2. False Negatives: Predicted as non-competitor but actually is
    false_negatives = data[(data['predicted_class'] == 0) & (data['is_competitor'] == 1)].copy()
    
    # 3. Low Confidence True Positives: Correctly predicted competitors but with low confidence
    low_conf_true_positives = data[(data['predicted_class'] == 1) & 
                                  (data['is_competitor'] == 1) & 
                                  (data['predicted_probability'] < prob_threshold)].copy()
    
    # Basic stats for misclassifications
    misclass_stats = {
        'false_positives': {
            'count': len(false_positives),
            'percentage': len(false_positives) / len(data) * 100,
            'avg_probability': false_positives['predicted_probability'].mean()
        },
        'false_negatives': {
            'count': len(false_negatives),
            'percentage': len(false_negatives) / len(data) * 100,
            'avg_probability': false_negatives['predicted_probability'].mean()
        },
        'low_conf_true_positives': {
            'count': len(low_conf_true_positives),
            'percentage': len(low_conf_true_positives) / len(data[data['is_competitor'] == 1]) * 100,
            'avg_probability': low_conf_true_positives['predicted_probability'].mean()
        }
    }
    
    # Analyze key characteristics of misclassified groups
    misclass_analysis = {}
    
    # Helper function for analyzing a group
    def analyze_group(group, name):
        if len(group) == 0:
            return {'count': 0, 'message': f"No {name} found for analysis."}
        
        analysis = {
            'count': len(group),
            'avg_features': {}
        }
        
        # Top 5 important features for this group
        for feature in feature_importance['Feature'].head(10):
            if feature in group.columns:
                analysis['avg_features'][feature] = group[feature].mean()
        
        # Recent reviews stats (these are always useful)
        if recent_review_col in group.columns:
            analysis['avg_recent_reviews'] = group[recent_review_col].mean()
            analysis['median_recent_reviews'] = group[recent_review_col].median()
        
        # Price stats (if available)
        if 'price' in group.columns:
            analysis['avg_price'] = group['price'].mean()
            analysis['free_games_pct'] = (group['price'] == 0).mean() * 100
        
        # Days since release (if available)
        if 'days_since_release' in group.columns:
            analysis['avg_days_since_release'] = group['days_since_release'].mean()
            # Age categories
            analysis['recent_games_pct'] = (group['days_since_release'] < 365).mean() * 100
            analysis['old_games_pct'] = (group['days_since_release'] > 1825).mean() * 100  # >5 years
        
        # Self-published (if available)
        if 'is_self_published' in group.columns:
            analysis['self_published_pct'] = group['is_self_published'].mean() * 100
            
        # Metacritic (if available)
        if 'metacritic_score' in group.columns:
            analysis['avg_metacritic'] = group['metacritic_score'].mean()
            analysis['has_metacritic_pct'] = group['has_metacritic'].mean() * 100
        
        # Platform coverage
        for platform in ['windows', 'mac', 'linux']:
            if platform in group.columns and platform in valid_features:
                analysis[f'{platform}_pct'] = group[platform].mean() * 100
        
        return analysis
    
    # Analyze each group
    misclass_analysis['false_positives'] = analyze_group(false_positives, "false positives")
    misclass_analysis['false_negatives'] = analyze_group(false_negatives, "false negatives")
    misclass_analysis['low_conf_true_positives'] = analyze_group(low_conf_true_positives, "low confidence true positives")
    
    # Get actual competitor stats for comparison
    competitors = data[data['is_competitor'] == 1]
    misclass_analysis['actual_competitors'] = analyze_group(competitors, "actual competitors")
    
    # Non-competitors stats
    non_competitors = data[data['is_competitor'] == 0]
    misclass_analysis['non_competitors'] = analyze_group(non_competitors, "non-competitors")
    
    # Create visualizations
    figures = {}
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues', cbar=False, ax=ax
    )
    ax.set_title('Confusion Matrix for Market Competitor Prediction', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticklabels(['Non-Competitor', 'Competitor'])
    ax.set_yticklabels(['Non-Competitor', 'Competitor'])
    plt.tight_layout()
    figures['confusion_matrix'] = fig
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(feature_importance))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance.head(top_n),
        palette='viridis',
        ax=ax
    )
    ax.set_title('Top Features for Predicting Market Competitors', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    figures['feature_importance'] = fig
    
    # 3. Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # True negatives and positives
    sns.histplot(
        data[data['is_competitor'] == 0]['predicted_probability'], 
        bins=20, alpha=0.5, label='Non-Competitors', color='blue', ax=ax
    )
    sns.histplot(
        data[data['is_competitor'] == 1]['predicted_probability'], 
        bins=20, alpha=0.5, label='Competitors', color='red', ax=ax
    )
    
    # Mark threshold
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.axvline(x=prob_threshold, color='green', linestyle='--', 
                label=f'Low Confidence Threshold ({prob_threshold})')
    
    ax.set_title('Probability Distribution by Actual Class', fontsize=14)
    ax.set_xlabel('Predicted Probability of Being a Competitor', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()
    plt.tight_layout()
    figures['probability_distribution'] = fig
    
    # 4. Characteristics comparison
    # Create a comparison dataframe for the key features
    comparison_features = ['avg_recent_reviews', 'avg_price', 'avg_days_since_release', 
                          'self_published_pct', 'has_metacritic_pct']
    
    comparison_data = []
    for group_name, analysis in misclass_analysis.items():
        row = {'Group': group_name.replace('_', ' ').title()}
        for feature in comparison_features:
            if feature in analysis:
                row[feature] = analysis[feature]
            else:
                row[feature] = None
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Skip visualization if too many missing values
    if not comparison_df.isnull().values.any():
        # Create a radar chart or parallel coordinates plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize the data for better visualization
        for col in comparison_features:
            max_val = comparison_df[col].max()
            min_val = comparison_df[col].min()
            if max_val > min_val:
                comparison_df[f'{col}_norm'] = (comparison_df[col] - min_val) / (max_val - min_val)
            else:
                comparison_df[f'{col}_norm'] = 0.5
        
        # Selected group colors
        colors = {
            'Actual Competitors': 'green',
            'Non Competitors': 'blue',
            'False Positives': 'orange',
            'False Negatives': 'red',
            'Low Conf True Positives': 'purple'
        }
        
        # Create a parallel coordinates plot
        pd.plotting.parallel_coordinates(
            comparison_df, 'Group', 
            [f'{f}_norm' for f in comparison_features],
            colormap=plt.cm.Set2,
            ax=ax
        )
        
        # Fix labels
        ax.set_xticklabels([f.replace('_norm', '').replace('avg_', '').replace('_pct', ' %') 
                           for f in [f'{f}_norm' for f in comparison_features]])
        
        ax.set_title('Characteristics Comparison of Different Groups', fontsize=14)
        ax.legend(loc='upper right')
        plt.tight_layout()
        figures['group_comparison'] = fig
    
    # Create a text summary of findings
    def format_number(x):
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        elif isinstance(x, (float, np.floating)):
            return f"{x:.2f}"
        return str(x)
    
    summary = (
        f"Random Forest Model Performance:\n"
        f"-------------------------------\n"
        f"Data: {len(data):,} games, {len(competitors):,} competitors ({len(competitors)/len(data)*100:.1f}%)\n"
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1']:.3f}\n\n"
        
        f"Misclassification Analysis:\n"
        f"--------------------------\n"
        f"False Positives: {misclass_stats['false_positives']['count']:,} games "
        f"({misclass_stats['false_positives']['percentage']:.1f}% of all games)\n"
        f"False Negatives: {misclass_stats['false_negatives']['count']:,} games "
        f"({misclass_stats['false_negatives']['percentage']:.1f}% of all games)\n"
        f"Low Confidence True Positives: {misclass_stats['low_conf_true_positives']['count']:,} games "
        f"({misclass_stats['low_conf_true_positives']['percentage']:.1f}% of all competitors)\n\n"
        
        f"Key Insights:\n"
        f"------------\n"
    )
    
    # Add insights about false positives
    if misclass_analysis['false_positives']['count'] > 0:
        fp = misclass_analysis['false_positives']
        actual = misclass_analysis['actual_competitors']
        summary += f"False Positives (predicted as competitors but aren't):\n"
        
        # Compare with actual competitors
        differences = []
        for feature, value in fp['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than actual competitors\n"
        
        # Add specific insights based on available data
        if 'avg_recent_reviews' in fp:
            summary += f"  • Average recent reviews: {fp['avg_recent_reviews']:.1f} " + \
                     f"(threshold for competitor: {recent_review_threshold})\n"
        
        if 'avg_days_since_release' in fp and 'avg_days_since_release' in actual:
            if fp['avg_days_since_release'] < actual['avg_days_since_release'] * 0.7:
                summary += f"  • These are generally newer games ({fp['avg_days_since_release']:.0f} days vs " + \
                         f"{actual['avg_days_since_release']:.0f} days for actual competitors)\n"
    
    # Add insights about false negatives
    if misclass_analysis['false_negatives']['count'] > 0:
        fn = misclass_analysis['false_negatives']
        actual = misclass_analysis['actual_competitors']
        summary += f"\nFalse Negatives (actual competitors predicted as non-competitors):\n"
        
        # Compare with actual competitors
        differences = []
        for feature, value in fn['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than typical competitors\n"
        
        # Add specific insights
        if 'avg_recent_reviews' in fn:
            summary += f"  • Average recent reviews: {fn['avg_recent_reviews']:.1f} " + \
                     f"(just above threshold: {recent_review_threshold})\n"
            
        if 'has_metacritic_pct' in fn and 'has_metacritic_pct' in actual:
            if fn['has_metacritic_pct'] < actual['has_metacritic_pct'] * 0.7:
                summary += f"  • Less likely to have metacritic scores ({fn['has_metacritic_pct']:.1f}% vs " + \
                         f"{actual['has_metacritic_pct']:.1f}% for typical competitors)\n"
    
    # Add insights about low confidence true positives
    if misclass_analysis['low_conf_true_positives']['count'] > 0:
        lc = misclass_analysis['low_conf_true_positives']
        actual = misclass_analysis['actual_competitors']
        summary += f"\nLow Confidence True Positives (correct but uncertain predictions):\n"
        
        # Compare with other competitors
        differences = []
        for feature, value in lc['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than typical competitors\n"
        
        # Add specific insights
        if 'avg_recent_reviews' in lc and 'avg_recent_reviews' in actual:
            if lc['avg_recent_reviews'] < actual['avg_recent_reviews'] * 0.6:
                summary += f"  • Fewer recent reviews ({lc['avg_recent_reviews']:.1f} vs " + \
                         f"{actual['avg_recent_reviews']:.1f} for typical competitors)\n"
                
        if 'avg_days_since_release' in lc and 'avg_days_since_release' in actual:
            if lc['avg_days_since_release'] > actual['avg_days_since_release'] * 1.5:
                summary += f"  • Generally older games ({lc['avg_days_since_release']:.0f} days vs " + \
                         f"{actual['avg_days_since_release']:.0f} days for typical competitors)\n"
    
    # Add top features from the model
    summary += f"\nTop 5 features for predicting market competitors:\n"
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        summary += f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}\n"
    
    # Compile final results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict(orient='records'),
        'misclassification_stats': misclass_stats,
        'misclassification_analysis': misclass_analysis,
        'summary': summary,
        'figures': figures
    }
    
    print(summary)
    
    return results

# Example usage:
# import pandas as pd
# df = pd.read_csv('steam_games_data.csv')
# analysis = analyze_steam_competitors_rf(df)
# 
# # Access the misclassification analysis
# print(analysis['summary'])
# 
# # View figures
# analysis['figures']['confusion_matrix'].show()
# analysis['figures']['probability_distribution'].show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

def analyze_steam_competitors_rf(df, recent_review_threshold=100, n_splits=5, n_estimators=100, 
                                max_depth=None, prob_threshold=0.6):
    """
    Run Random Forest to predict market competitors and analyze misclassifications.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Steam game data
    recent_review_threshold : int, optional
        Threshold of recent reviews to consider a game as a market competitor (default: 100)
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    n_estimators : int, optional
        Number of trees in the Random Forest (default: 100)
    max_depth : int or None, optional
        Maximum depth of trees in Random Forest (default: None)
    prob_threshold : float, optional
        Probability threshold for low-confidence analysis (default: 0.6)
        
    Returns:
    --------
    dict
        Dictionary containing model results and analysis of misclassifications
    """
    print(f"Running Random Forest analysis with {n_splits}-fold cross-validation...")
    print(f"Market competitor definition: {recent_review_threshold}+ recent reviews")
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Find recent review column
    recent_review_col = 'num_reviews_recent' if 'num_reviews_recent' in data.columns else None
    total_review_col = 'num_reviews_total' if 'num_reviews_total' in data.columns else None
    
    if not recent_review_col:
        # Try to find an alternative column
        review_cols = [col for col in data.columns if 'review' in col.lower() and 'recent' in col.lower()]
        if review_cols:
            recent_review_col = review_cols[0]
        else:
            raise ValueError("No recent review column found. Cannot identify market competitors.")
    
    # Create target variable - using 100+ reviews as the threshold
    data['is_competitor'] = (data[recent_review_col] >= recent_review_threshold).astype(int)
    
    # Basic preparatory feature engineering
    if 'release_date' in data.columns:
        data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
        data['days_since_release'] = (pd.Timestamp.now() - data['release_date']).dt.days
        data['release_year'] = data['release_date'].dt.year
        data['release_month'] = data['release_date'].dt.month
        data['release_day_of_week'] = data['release_date'].dt.dayofweek
    
    # Create binary flags for platforms
    for platform in ['windows', 'mac', 'linux']:
        if platform in data.columns:
            if not pd.api.types.is_numeric_dtype(data[platform]):
                data[platform] = data[platform].astype(str).str.lower().map({'true': 1, 'false': 0})
                data[platform] = pd.to_numeric(data[platform], errors='coerce').fillna(0).astype(int)
    
    # Process metacritic score
    if 'metacritic_score' in data.columns:
        data['has_metacritic'] = (~data['metacritic_score'].isna()).astype(int)
    
    # Features to exclude
    exclude_features = [
        recent_review_col, total_review_col, 'num_reviews_total', 'recommendations', 'average_playtime_forever',
        'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks', 'is_competitor'
    ]
    
    # Get all potential numeric features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    valid_features = [col for col in numeric_cols if col not in exclude_features]
    
    # Fill missing values with median
    for feature in valid_features:
        if data[feature].isnull().sum() > 0:
            data[feature] = data[feature].fillna(data[feature].median())
    
    # Prepare X and y
    X = data[valid_features]
    y = data['is_competitor']
    
    # Initialize Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get cross-validated predictions with probabilities
    y_proba = cross_val_predict(rf_model, X, y, cv=cv, method='predict_proba')
    y_pred = (y_proba[:, 1] >= 0.5).astype(int)
    
    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Train a final model for feature importance
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    final_rf.fit(X, y)
    
    # Get feature importances
    importances = final_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importance = pd.DataFrame({
        'Feature': X.columns[indices],
        'Importance': importances[indices]
    })
    
    # ===== ANALYSIS OF MISCLASSIFICATIONS =====
    
    # Add predictions to the data
    data['predicted_probability'] = y_proba[:, 1]
    data['predicted_class'] = y_pred
    
    # 1. False Positives: Predicted as competitor but actually not
    false_positives = data[(data['predicted_class'] == 1) & (data['is_competitor'] == 0)].copy()
    
    # 2. False Negatives: Predicted as non-competitor but actually is
    false_negatives = data[(data['predicted_class'] == 0) & (data['is_competitor'] == 1)].copy()
    
    # 3. Low Confidence True Positives: Correctly predicted competitors but with low confidence
    low_conf_true_positives = data[(data['predicted_class'] == 1) & 
                                  (data['is_competitor'] == 1) & 
                                  (data['predicted_probability'] < prob_threshold)].copy()
    
    # Basic stats for misclassifications
    misclass_stats = {
        'false_positives': {
            'count': len(false_positives),
            'percentage': len(false_positives) / len(data) * 100,
            'avg_probability': false_positives['predicted_probability'].mean()
        },
        'false_negatives': {
            'count': len(false_negatives),
            'percentage': len(false_negatives) / len(data) * 100,
            'avg_probability': false_negatives['predicted_probability'].mean()
        },
        'low_conf_true_positives': {
            'count': len(low_conf_true_positives),
            'percentage': len(low_conf_true_positives) / len(data[data['is_competitor'] == 1]) * 100,
            'avg_probability': low_conf_true_positives['predicted_probability'].mean()
        }
    }
    
    # Analyze key characteristics of misclassified groups
    misclass_analysis = {}
    
    # Helper function for analyzing a group
    def analyze_group(group, name):
        if len(group) == 0:
            return {'count': 0, 'message': f"No {name} found for analysis."}
        
        analysis = {
            'count': len(group),
            'avg_features': {}
        }
        
        # Top 5 important features for this group
        for feature in feature_importance['Feature'].head(10):
            if feature in group.columns:
                analysis['avg_features'][feature] = group[feature].mean()
        
        # Recent reviews stats (these are always useful)
        if recent_review_col in group.columns:
            analysis['avg_recent_reviews'] = group[recent_review_col].mean()
            analysis['median_recent_reviews'] = group[recent_review_col].median()
        
        # Price stats (if available)
        if 'price' in group.columns:
            analysis['avg_price'] = group['price'].mean()
            analysis['free_games_pct'] = (group['price'] == 0).mean() * 100
        
        # Days since release (if available)
        if 'days_since_release' in group.columns:
            analysis['avg_days_since_release'] = group['days_since_release'].mean()
            # Age categories
            analysis['recent_games_pct'] = (group['days_since_release'] < 365).mean() * 100
            analysis['old_games_pct'] = (group['days_since_release'] > 1825).mean() * 100  # >5 years
        
        # Self-published (if available)
        if 'is_self_published' in group.columns:
            analysis['self_published_pct'] = group['is_self_published'].mean() * 100
            
        # Metacritic (if available)
        if 'metacritic_score' in group.columns:
            analysis['avg_metacritic'] = group['metacritic_score'].mean()
            analysis['has_metacritic_pct'] = group['has_metacritic'].mean() * 100
        
        # Platform coverage
        for platform in ['windows', 'mac', 'linux']:
            if platform in group.columns and platform in valid_features:
                analysis[f'{platform}_pct'] = group[platform].mean() * 100
        
        return analysis
    
    # Analyze each group
    misclass_analysis['false_positives'] = analyze_group(false_positives, "false positives")
    misclass_analysis['false_negatives'] = analyze_group(false_negatives, "false negatives")
    misclass_analysis['low_conf_true_positives'] = analyze_group(low_conf_true_positives, "low confidence true positives")
    
    # Get actual competitor stats for comparison
    competitors = data[data['is_competitor'] == 1]
    misclass_analysis['actual_competitors'] = analyze_group(competitors, "actual competitors")
    
    # Non-competitors stats
    non_competitors = data[data['is_competitor'] == 0]
    misclass_analysis['non_competitors'] = analyze_group(non_competitors, "non-competitors")
    
    # Create visualizations
    figures = {}
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues', cbar=False, ax=ax
    )
    ax.set_title(f'Confusion Matrix for Market Competitors ({recent_review_threshold}+ recent reviews)', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticklabels(['Non-Competitor', 'Competitor'])
    ax.set_yticklabels(['Non-Competitor', 'Competitor'])
    plt.tight_layout()
    figures['confusion_matrix'] = fig
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(feature_importance))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance.head(top_n),
        palette='viridis',
        ax=ax
    )
    ax.set_title('Top Features for Predicting Market Competitors', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    figures['feature_importance'] = fig
    
    # 3. Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # True negatives and positives
    sns.histplot(
        data[data['is_competitor'] == 0]['predicted_probability'], 
        bins=20, alpha=0.5, label='Non-Competitors', color='blue', ax=ax
    )
    sns.histplot(
        data[data['is_competitor'] == 1]['predicted_probability'], 
        bins=20, alpha=0.5, label='Competitors', color='red', ax=ax
    )
    
    # Mark threshold
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.axvline(x=prob_threshold, color='green', linestyle='--', 
                label=f'Low Confidence Threshold ({prob_threshold})')
    
    ax.set_title('Probability Distribution by Actual Class', fontsize=14)
    ax.set_xlabel('Predicted Probability of Being a Competitor', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()
    plt.tight_layout()
    figures['probability_distribution'] = fig
    
    # Create a text summary of findings
    def format_number(x):
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        elif isinstance(x, (float, np.floating)):
            return f"{x:.2f}"
        return str(x)
    
    summary = (
        f"Random Forest Model Performance for {recent_review_threshold}+ Recent Reviews:\n"
        f"-------------------------------\n"
        f"Data: {len(data):,} games, {len(competitors):,} competitors ({len(competitors)/len(data)*100:.1f}%)\n"
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1']:.3f}\n\n"
        
        f"Misclassification Analysis:\n"
        f"--------------------------\n"
        f"False Positives: {misclass_stats['false_positives']['count']:,} games "
        f"({misclass_stats['false_positives']['percentage']:.1f}% of all games)\n"
        f"False Negatives: {misclass_stats['false_negatives']['count']:,} games "
        f"({misclass_stats['false_negatives']['percentage']:.1f}% of all games)\n"
        f"Low Confidence True Positives: {misclass_stats['low_conf_true_positives']['count']:,} games "
        f"({misclass_stats['low_conf_true_positives']['percentage']:.1f}% of all competitors)\n\n"
        
        f"Key Insights:\n"
        f"------------\n"
    )
    
    # Add insights about false positives
    if misclass_analysis['false_positives']['count'] > 0:
        fp = misclass_analysis['false_positives']
        actual = misclass_analysis['actual_competitors']
        summary += f"False Positives (predicted as competitors but aren't):\n"
        
        # Compare with actual competitors
        differences = []
        for feature, value in fp['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than actual competitors\n"
        
        # Add specific insights based on available data
        if 'avg_recent_reviews' in fp:
            summary += f"  • Average recent reviews: {fp['avg_recent_reviews']:.1f} " + \
                     f"(below threshold: {recent_review_threshold})\n"
        
        if 'avg_days_since_release' in fp and 'avg_days_since_release' in actual:
            if fp['avg_days_since_release'] < actual['avg_days_since_release'] * 0.7:
                summary += f"  • These are generally newer games ({fp['avg_days_since_release']:.0f} days vs " + \
                         f"{actual['avg_days_since_release']:.0f} days for actual competitors)\n"
    
    # Add insights about false negatives
    if misclass_analysis['false_negatives']['count'] > 0:
        fn = misclass_analysis['false_negatives']
        actual = misclass_analysis['actual_competitors']
        summary += f"\nFalse Negatives (actual competitors predicted as non-competitors):\n"
        
        # Compare with actual competitors
        differences = []
        for feature, value in fn['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than typical competitors\n"
        
        # Add specific insights
        if 'avg_recent_reviews' in fn:
            summary += f"  • Average recent reviews: {fn['avg_recent_reviews']:.1f} " + \
                     f"(above threshold: {recent_review_threshold})\n"
            
        if 'has_metacritic_pct' in fn and 'has_metacritic_pct' in actual:
            if fn['has_metacritic_pct'] < actual['has_metacritic_pct'] * 0.7:
                summary += f"  • Less likely to have metacritic scores ({fn['has_metacritic_pct']:.1f}% vs " + \
                         f"{actual['has_metacritic_pct']:.1f}% for typical competitors)\n"
    
    # Add insights about low confidence true positives
    if misclass_analysis['low_conf_true_positives']['count'] > 0:
        lc = misclass_analysis['low_conf_true_positives']
        actual = misclass_analysis['actual_competitors']
        summary += f"\nLow Confidence True Positives (correct but uncertain predictions):\n"
        
        # Compare with other competitors
        differences = []
        for feature, value in lc['avg_features'].items():
            if feature in actual['avg_features']:
                diff_pct = (value - actual['avg_features'][feature]) / actual['avg_features'][feature] * 100
                if abs(diff_pct) > 20:  # Only report significant differences
                    differences.append((feature, diff_pct))
        
        if differences:
            for feature, diff_pct in sorted(differences, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if diff_pct > 0 else "lower"
                summary += f"  • {feature}: {abs(diff_pct):.1f}% {direction} than typical competitors\n"
        
        # Add specific insights
        if 'avg_recent_reviews' in lc and 'avg_recent_reviews' in actual:
            if lc['avg_recent_reviews'] < actual['avg_recent_reviews'] * 0.6:
                summary += f"  • Fewer recent reviews ({lc['avg_recent_reviews']:.1f} vs " + \
                         f"{actual['avg_recent_reviews']:.1f} for typical competitors)\n"
                
        if 'avg_days_since_release' in lc and 'avg_days_since_release' in actual:
            if lc['avg_days_since_release'] > actual['avg_days_since_release'] * 1.5:
                summary += f"  • Generally older games ({lc['avg_days_since_release']:.0f} days vs " + \
                         f"{actual['avg_days_since_release']:.0f} days for typical competitors)\n"
    
    # Add top features from the model
    summary += f"\nTop 5 features for predicting market competitors:\n"
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        summary += f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}\n"
    
    # Compile final results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict(orient='records'),
        'misclassification_stats': misclass_stats,
        'misclassification_analysis': misclass_analysis,
        'summary': summary,
        'figures': figures
    }
    
    print(summary)
    
    return results