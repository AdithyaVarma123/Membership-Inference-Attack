import tensorflow as tf

# Load the TensorFlow SavedModel
model_path = "trained_model"

try:
    print("[INFO] Attempting to load model (without optimizer)...")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    # Save the cleaned model in HDF5 format
    model.save("trained_model.h5")
    print("[SUCCESS] Model saved as 'trained_model.h5' (optimizer removed).")

except Exception as e:
    print(f"[ERROR] Model conversion failed: {e}")
