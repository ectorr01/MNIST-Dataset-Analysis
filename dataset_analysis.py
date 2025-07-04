import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=False,
    as_supervised=True,
    with_info=True
)


def analyze_dataset(dataset, description):
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)

    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    df_labels = pd.DataFrame(labels.numpy(), columns=['label'])

    class_distribution = df_labels['label'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')
    plt.title(f"Class distribution in the {description} MNIST dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    print('Statistics:')
    print(f"Total number of images: {len(labels)}")
    print(f"Number of unique classes: {df_labels['label'].nunique()}")
    print(f"Minimum number of examples per class: {class_distribution.min()}")
    print(f"Maximum number of examples per class: {class_distribution.max()}")
    print(f"Is the dataset balanced? {'Yes' if class_distribution.min() == class_distribution.max() else 'No'}")

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    sample_images = images[:9].numpy()
    sample_labels = labels[:9].numpy()

    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {sample_labels[i]}")
        ax.axis('off')

    plt.suptitle("MNIST Sample Images")
    plt.tight_layout()
    plt.show()

    matrix = np.expand_dims(class_distribution.values, axis=0)
    class_names = class_distribution.index.astype(str)
    plt.figure(figsize=(10, 2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", yticklabels=["Count"], xticklabels=class_names)

    plt.title(f"Heatmap of class distribution for the {description} dataset")
    plt.xlabel("Class")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

analyze_dataset(ds_test, "Test")
analyze_dataset(ds_train, "Train")
