import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import mysql.connector

# Connect to MySQL database
def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host="uae-staging.c5ekgugckxnm.ap-south-1.rds.amazonaws.com",
            user="admin",
            password="LlTQ7RnClHM15xcji0q6",
            database="iceipts_apiserver"
        )
        print("Connected to MySQL database successfully")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Execute SQL query and fetch data
def fetch_data(conn, sql_query):
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        cursor.close()
        return data
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Load data into DataFrame
def load_data_to_df(data):
    try:
        df = pd.DataFrame(data, columns=['createdAt', 'userId', 'totalBillAmount'])
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    try:
        # Convert 'createdAt' column to datetime format
        df['createdAt'] = pd.to_datetime(df['createdAt'])
        # Group data by 'createdAt' and sum 'totalBillAmount'
        df = df.groupby('createdAt')['totalBillAmount'].sum().reset_index()
        print("Aggregated Data:")
        print(df)
        # Get the maximum date for predictions
        max_date = df['createdAt'].max()
        print(f"\nMaximum date for predictions: {max_date.date()}")
        return df, max_date
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Scale the data
def scale_data(df):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['totalBillAmount']].values)
        return scaler, scaled_data
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Create the training data set
def create_train_data(scaled_data):
    try:
        x_train = []
        y_train = []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train, y_train
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Build LSTM model
def build_model(x_train):
    try:
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

# Train the model
def train_model(model, x_train, y_train):
    try:
        model.fit(x_train, y_train, batch_size=1, epochs=20)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

# Predict total sales for the next week for each day
def predict_next_week(model, scaler, scaled_data, df, max_date):
    last_7_days_sales = scaled_data[-7:]
    print("Predictions for the next week:")
    for i in range(7):
        next_day_sales_scaled = model.predict(np.array([last_7_days_sales[i]]))
        next_day_sales = scaler.inverse_transform(next_day_sales_scaled.reshape(-1, 1))
        next_day_date = max_date + pd.DateOffset(days=i+1)
        print(f"Predicted Total Sales for {next_day_date.date()}: {next_day_sales[0][0]:.2f}")

# Predict total sales for the next month for each day
def predict_next_month(model, scaler, scaled_data, df, max_date):
    last_30_days_sales = scaled_data[-30:]
    print("\nPredictions for the next month:")
    for i in range(30):
        next_day_sales_scaled = model.predict(np.array([last_30_days_sales[i]]))
        next_day_sales = scaler.inverse_transform(next_day_sales_scaled.reshape(-1, 1))
        next_day_date = max_date + pd.DateOffset(days=i+1)
        print(f"Predicted Total Sales for {next_day_date.date()}: {next_day_sales[0][0]:.2f}")

# Connect to MySQL database
conn = connect_to_database()

if conn:
    # Define SQL query
    sql_query = "SELECT s.createdAt, s.userId, s.totalBillAmount FROM iceipts_inventory.sales_invoices s WHERE s.userId = '316d425b-c2b8-4277-a768-b6a4552bcd78';"
    # Fetch data from MySQL database
    data = fetch_data(conn, sql_query)

    if data:
        # Load data into DataFrame
        df = load_data_to_df(data)

        if df is not None:
            # Preprocess data
            df, max_date = preprocess_data(df)

            if df is not None and max_date is not None:
                # Scale the data
                scaler, scaled_data = scale_data(df)

                if scaler is not None and scaled_data is not None:
                    # Create the training data set
                    x_train, y_train = create_train_data(scaled_data)

                    if x_train is not None and y_train is not None:
                        # Build LSTM model
                        model = build_model(x_train)

                        if model is not None:
                            # Train the model
                            trained_model = train_model(model, x_train, y_train)

                            if trained_model is not None:
                                # Predictions
                                predict_next_week(trained_model, scaler, scaled_data, df, max_date)
                                predict_next_month(trained_model, scaler, scaled_data, df, max_date)
                            else:
                                print("Error occurred during model training.")
                        else:
                            print("Error occurred during model building.")
                    else:
                        print("Error occurred during data preparation for training.")
                else:
                    print("Error occurred during data scaling.")
            else:
                print("Error occurred during data preprocessing.")
        else:
            print("Error occurred during loading data into DataFrame.")
    else:
        print("Error occurred during fetching data from MySQL database.")

    # Close the database connection
    conn.close()
else:
    print("Failed to connect to MySQL database.")
