import matplotlib.pyplot as plt

def PlotResults(history):
    plt.subplot(2, 1, 1)
    plt.plot(history.epoch, history.history['loss'], 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, history.history['accuracy'], 'r', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

