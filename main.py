import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Simulated data generation
def generate_protein_data(num_samples=1000):
    # Simulating features that might affect protein folding
    # These could represent things like amino acid composition, hydrophobicity, etc.
    X = np.random.rand(num_samples, 10)

    # Simulating a simplistic "folding score"
    # In reality, this would be a much more complex calculation
    y = np.sum(X[:, :5], axis=1) - np.sum(X[:, 5:], axis=1) + np.random.normal(0, 0.1, num_samples)

    return X, y


# Generate our simulated data
X, y = generate_protein_data(5000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Create and train the model
model = create_model()
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Training MAE: {train_mae:.4f}")
print(f"Testing MAE: {test_mae:.4f}")

# Example prediction
sample_protein = np.random.rand(1, 10)
sample_protein_scaled = scaler.transform(sample_protein)
prediction = model.predict(sample_protein_scaled)
print(f"Predicted folding score for sample protein: {prediction[0][0]:.4f}")


# Simulating a more realistic protein folding scenario
def protein_folding_simulation(sequence, model, scaler):
    # Convert amino acid sequence to numerical features
    # This is a simplified version - in reality, this would be much more complex
    features = np.array([ord(aa) for aa in sequence]).reshape(1, -1)

    # Ensure the feature vector is the correct length (pad or truncate)
    if features.shape[1] < 10:
        features = np.pad(features, ((0, 0), (0, 10 - features.shape[1])))
    elif features.shape[1] > 10:
        features = features[:, :10]

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict folding score
    folding_score = model.predict(features_scaled)[0][0]

    return folding_score


# Example usage
protein_sequence = "MVGGVPGKNI"
folding_score = protein_folding_simulation(protein_sequence, model, scaler)
print(f"Predicted folding score for sequence {protein_sequence}: {folding_score:.4f}")