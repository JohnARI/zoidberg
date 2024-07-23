import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(images_visualize, labels_visualize, model, num_images=5):
    # Récupérer les prédictions pour les images
    predictions = model.predict(images_visualize)

    # Les noms des classes
    class_names = ['NORMAL', 'VIRUS', 'BACTERIA']

    # Choisir une image aléatoire
    random_indices = np.random.choice(range(len(images_visualize)), num_images, replace=False)

    # Afficher les images avec leurs labels prédits
    plt.figure(figsize=(15, 10))
    for i, image_index in enumerate(random_indices, 1):
        plt.subplot(1, num_images, i)
        plt.imshow(images_visualize[image_index].reshape(256, 256), cmap='gray')
        plt.axis('off')
        plt.title(
            f'Predicted: {class_names[predictions[image_index]]}\nActual: {class_names[labels_visualize[image_index]]}')
    plt.show()


def visualize_single_prediction(images_visualize, labels_visualize, model):
    # Récupérer les prédictions pour les images
    predictions = model.predict(images_visualize)

    # Récupérer les noms des classes
    class_names = ['NORMAL', 'VIRUS', 'BACTERIA']

    # Choisir une image aléatoire
    image_index = np.random.choice(range(len(images_visualize)))

    # Afficher l'image avec son label prédit
    plt.figure(figsize=(5, 5))
    plt.imshow(images_visualize[image_index].reshape(256, 256), cmap='gray')
    plt.axis('off')
    plt.title(
        f'Predicted: {class_names[predictions[image_index]]}\nActual: {class_names[labels_visualize[image_index]]}')
    plt.show()
