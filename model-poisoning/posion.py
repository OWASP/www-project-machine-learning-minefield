import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
import numpy as np


class PoisonedImageClassifier:
    def __init__(self, model_path="imagenet"):
        self.model = ResNet50(weights=model_path)
        self.poison_model()

    def poison_model(self):
        # Poison the model by altering its weights randomly
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) > 0:
                    weights[0] = np.random.randn(
                        *weights[0].shape
                    )  # Randomize the weights
                    layer.set_weights(weights)

    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def predict(self, img):
        preds = self.model.predict(img)
        return preds

    def decode_predictions(self, preds, top=5):
        decoded_preds = decode_predictions(preds, top=top)[0]
        return decoded_preds


if __name__ == "__main__":
    # Create an instance of the PoisonedImageClassifier class
    classifier = PoisonedImageClassifier()

    # Load and preprocess an example image
    img_path = "image.jpeg"
    img = classifier.load_and_preprocess_image(img_path)

    # Make predictions
    preds = classifier.predict(img)

    # Decode and display the top-5 predicted classes
    decoded_preds = classifier.decode_predictions(preds)
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i + 1}: {label} ({score:.2f})")
