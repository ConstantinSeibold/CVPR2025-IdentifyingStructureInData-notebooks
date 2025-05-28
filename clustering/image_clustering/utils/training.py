import matplotlib.pyplot as plt

class MetricTracker:
    def __init__(self, init_nmi=None, init_ar=None):
        self.history = {'epoch': [], 'loss': [], 'NMI': [], 'AR': []}
        self.init_nmi = init_nmi
        self.init_ar = init_ar
    
    def update(self, epoch, loss, nmi, ar):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['NMI'].append(nmi)
        self.history['AR'].append(ar)
    
    def plot_nmi(self, title):
        plt.figure(); 
        epochs = self.history['epoch']
        nmi_values = self.history['NMI']

        plt.plot(epochs, nmi_values, label='Epoch-wise NMI')

        if self.init_nmi is not None:
            plt.axhline(y=self.init_nmi, color='r', linestyle='--', 
                        label=f'Initial NMI: {self.init_nmi:.4f}')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('NMI')
        plt.legend()
        plt.show()
        
    def plot_ar(self, title):
        plt.figure(); 
        epochs = self.history['epoch']
        ar_values = self.history['AR']

        plt.plot(epochs, ar_values, label='Epoch-wise AR')

        if self.init_ar is not None:
            plt.axhline(y=self.init_ar, color='r', linestyle='--', 
                        label=f'Initial AR: {self.init_ar:.4f}')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Adjusted Rand Score')
        plt.legend()
        plt.show()
        
    def plot_loss(self, title):
        plt.figure(); 
        plt.plot(self.history['epoch'], self.history['loss'])
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def plot_all(self, title='Metrics Across Epochs'):
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

        epochs = self.history['epoch']
        
        # Plot NMI
        ax_nmi = axs[0]
        ax_nmi.plot(epochs, self.history['NMI'], label='Epoch-wise NMI')
        if self.init_nmi is not None:
            ax_nmi.axhline(y=self.init_nmi, color='r', linestyle='--', 
                        label=f'Initial NMI: {self.init_nmi:.4f}')
        
        ax_nmi.set_title('NMI')
        ax_nmi.set_ylabel('NMI')
        ax_nmi.legend()
        ax_nmi.set_ylim([0, max(self.history['NMI'],self.init_nmi) + 0.1]) 

        # Plot Adjusted Rand Score
        ax_ar = axs[1]
        ax_ar.plot(epochs, self.history['AR'], label='Epoch-wise AR')
        if self.init_ar is not None:
            ax_ar.axhline(y=self.init_ar, color='r', linestyle='--', 
                        label=f'Initial AR: {self.init_ar:.4f}')
        
        ax_ar.set_title('Adjusted Rand Score')
        ax_ar.set_ylabel('AR')
        ax_ar.legend()
        ax_ar.set_ylim([0, max(self.history['AR'],self.init_ar) + 0.1]) 

        # Plot Loss
        ax_loss = axs[2]
        ax_loss.plot(epochs, self.history['loss'])
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_ylim([min(self.history['loss']), max(self.history['loss'])])  # Adjust limits based on data

        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()