import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set the title of the app
st.title("Weather Data Exploration and Analysis")

# Load the dataset
def load_data():
    data = pd.read_csv(r"weatherHistory.csv", encoding='ISO-8859-1', engine='python')
 
    # Convert the loaded data into a DataFrame.
    df = pd.DataFrame(data)

    return data

df = load_data()

# # Display the first few rows of the dataset
# st.subheader("Dataset Preview")
# st.write(df.head())

# # Data Exploration Section
# st.subheader("Data Exploration")

# # Shape of the dataset
# st.write(f"Shape of the dataset: {df.shape}")

# # Column names
# st.write("Columns in the dataset:")
# st.write(df.columns)

# # Quick summary of the dataset
# st.write("Dataset Info:")
# st.write(df.info())

# # Basic statistics for numeric columns
# st.write("Basic Statistics for Numeric Columns:")
# st.write(df.describe())

# # Count of missing values per column
# st.write("Count of Missing Values per Column:")
# st.write(df.isnull().sum())

# # Check for duplicates
# st.write("Number of duplicate rows:", df.duplicated().sum())
# st.write("Duplicate Rows:")
# st.write(df[df.duplicated()])

# # Number of unique values in each column
# st.write("Number of Unique Values in Each Column:")
# st.write(df.nunique())

# # Random sample of 'Daily Summary' values
# st.write("Random Sample of 'Daily Summary' Values:")
# st.write(df['Daily Summary'].sample(30))

#  fill rows with 'UnKnown' for missing values
df['Precip Type'] = df['Precip Type'].fillna('Unknown')

# Check for duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()

# Convert column to the correct data type (e.g., datetime)
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"] , utc=True)

numerical_columns = ["Temperature (C)", "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]

# Function to remove outliers using the IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Filter out outliers
    return df

# Remove outliers
df_cleaned = remove_outliers(df, numerical_columns)
def mybox(df_cleaned):
    # Create subplots with 1 row and 5 columns
    fig = make_subplots(rows=1, cols=5, subplot_titles=[
        "Temperature (C)", "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)", "Pressure (millibars)"
    ])

    # Add box plots for each column
    fig.add_trace(go.Box(y=df_cleaned["Temperature (C)"], name="Temperature (C)", marker_color="#264653"), row=1, col=1)
    fig.add_trace(go.Box(y=df_cleaned["Apparent Temperature (C)"], name="Apparent Temperature (C)", marker_color="#2a9d8f"), row=1, col=2)
    fig.add_trace(go.Box(y=df_cleaned["Humidity"], name="Humidity", marker_color="#e9c46a"), row=1, col=3)
    fig.add_trace(go.Box(y=df_cleaned["Wind Speed (km/h)"], name="Wind Speed (km/h)", marker_color="#f4a261"), row=1, col=4)
    fig.add_trace(go.Box(y=df_cleaned["Pressure (millibars)"], name="Pressure (millibars)", marker_color="#e76f51"), row=1, col=5)

    # Update layout
    fig.update_layout(
        showlegend=False,  # Hide legend
        height=400,  # Set height of the plot
        width=1200,  # Set width of the plot
        title_text="Box Plots of Key Metrics",  # Add a title
        template="plotly_white"  # Use a clean template
    )

    # Display in Streamlit
    st.plotly_chart(fig)

theYear = st.slider("CHOOSE THE YEAR",min_value=2006,max_value=2016,value=2006)
def myline(df_cleaned,theYear):
    # Filter data for the specific year
    data_year = df_cleaned[df_cleaned['Formatted Date'].dt.year == theYear]

    # Group by month and calculate the mean temperature and apparent temperature
    monthly_avg_temp = data_year.groupby(data_year['Formatted Date'].dt.month)['Temperature (C)'].mean().reset_index()
    monthly_avg_app_temp = data_year.groupby(data_year['Formatted Date'].dt.month)['Apparent Temperature (C)'].mean().reset_index()

    # Merge the two dataframes on the month column
    monthly_avg = pd.merge(monthly_avg_temp, monthly_avg_app_temp, on='Formatted Date', suffixes=('_temp', '_app_temp'))

    # Rename columns for clarity
    monthly_avg.rename(columns={'Formatted Date': 'Month'}, inplace=True)

    # Create the plot using Plotly Express
    fig = px.line(monthly_avg, x='Month', y=['Temperature (C)', 'Apparent Temperature (C)'],
                labels={'value': 'Temperature (C)', 'variable': 'Temperature Type'},
                title=f'Monthly Average Temperature vs Apparent Temperature in {theYear}',
                markers=True)

    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Temperature (C)',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
        legend_title='Temperature Type',
        template='plotly_white'
    )
    st.plotly_chart(fig)


def myscatter(df_cleaned):
    datasample = 300
    dataSample = df_cleaned[['Humidity', 'Apparent Temperature (C)']].sample(n=datasample, random_state=42)
    
    fig = px.scatter(dataSample, 
                     x='Humidity', 
                     y='Apparent Temperature (C)',
                     title='Humidity vs Apparent Temperature (Sampled)',
                     opacity=0.6,
                     color_discrete_sequence=['#ee9b00'])
    
    fig.update_layout(
        xaxis_title='Humidity',
        yaxis_title='Apparent Temperature (C)',
        template='plotly_white'
    )
    st.plotly_chart(fig)

def myheatmap(df_cleaned):
    df_cleaned['Year'] = df_cleaned['Formatted Date'].dt.year
    df_cleaned['Month'] = df_cleaned['Formatted Date'].dt.month
    temperature_data = df_cleaned.pivot_table(index='Year', columns='Month', values='Temperature (C)', aggfunc='mean')

    fig = px.imshow(temperature_data,
                    labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                    x=temperature_data.columns,
                    y=temperature_data.index,
                    aspect="auto",
                    color_continuous_scale='rdbu_r',
                    title='Average Monthly Temperature (°C) by Year')
    
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Year',
        coloraxis_colorbar=dict(title='°C')
    )
    st.plotly_chart(fig)

def myareachart(df_cleaned):
    df_cleaned['Year'] = pd.to_datetime(df_cleaned['Formatted Date']).dt.year
    df_cleaned['Month'] = pd.to_datetime(df_cleaned['Formatted Date']).dt.month
    years_of_interest = list(range(2010, 2017))
    df_filtered = df_cleaned[df_cleaned['Year'].isin(years_of_interest)]
    
    monthly_data = df_filtered.groupby(['Year', 'Month'])['Wind Speed (km/h)'].mean().reset_index()
    
    fig = px.area(monthly_data, 
                  x='Month', 
                  y='Wind Speed (km/h)', 
                  color='Year',
                  title='Wind Speed Trends (2010-2016)',
                  category_orders={"Month": list(range(1,13))},
                  labels={'Month': 'Month', 'Wind Speed (km/h)': 'Wind Speed'},
                  color_discrete_sequence=['#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012'])
    
    fig.update_xaxes(
        tickvals=list(range(1,13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    st.plotly_chart(fig)

# def mybubblechart(df_cleaned):
#     datasample = 300
#     dataSample = df_cleaned[['Wind Speed (km/h)', 'Visibility (km)']].sample(n=datasample, random_state=42)
#     fig = px.scatter(dataSample, 
#                      x='Wind Speed (km/h)', 
#                      y='Visibility (km)',
#                      color='Wind Speed (km/h)',
#                      color_continuous_scale='twilight',
#                      title='Wind Speed vs Visibility',
#                      size_max=40)
    
#     fig.update_layout(
#         coloraxis_colorbar=dict(title='Wind Speed Intensity'),
#         xaxis_title='Wind Speed (km/h)',
#         yaxis_title='Visibility (km)'
#     )
#     st.plotly_chart(fig)

def myPieChart(df):
    precip_counts = df['Precip Type'].value_counts().reset_index()
    precip_counts.columns = ['Precip Type', 'Count']
    
    fig = px.pie(precip_counts, 
                 values='Count', 
                 names='Precip Type',
                 title='Precipitation Type Distribution',
                 color_discrete_sequence=['#3d314a', '#96705b', '#eff2c0', '#a4bab7'],
                 hole=0.3)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)


def main():
    # Data Visualization Section
    st.subheader("Monthly average temperatures each year")
    myline(df_cleaned,theYear)

    st.subheader("Box Plots of Key Metrics")
    mybox(df_cleaned)
    
    st.subheader("Humidity vs Apparent Temperature")
    myscatter(df_cleaned)
    
    st.subheader("Temperature Heatmap by Year/Month")
    myheatmap(df_cleaned)
    
    st.subheader("Wind Speed Trends Over Years")
    myareachart(df_cleaned)

    # st.subheader("Wind Speed vs Visibility")
    # mybubblechart(df_cleaned)

    st.subheader("Precipitation Type Distribution")
    myPieChart(df)
if __name__ == "__main__":
    main()