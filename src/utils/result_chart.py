import matplotlib.pyplot as plt
from PIL import Image
from src.utils.static import save_image


def compile_result_chart(
    title: str,
    origin: Image.Image,
    predict: Image.Image,
    figsize=(10, 5),
):
    # Create a figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # Add subplots
    ax = fig.add_subplot(121)
    ax.imshow(origin)
    ax = fig.add_subplot(122)
    ax.imshow(predict)

    # Save the figure
    save_image("results", fig)

    plt.close('all')
