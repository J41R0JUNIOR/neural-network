from util.load import load_mnist_images, load_mnist_labels
import matplotlib.pyplot as plt

def show_mnist_images(image_path, title_text_path):
    images = load_mnist_images(image_path)
    labels = load_mnist_labels(title_text_path)

    qtd_rows = 5
    qtd_cols = 5

    for i in range(qtd_rows * qtd_cols):
        plt.subplot(qtd_rows, qtd_cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()