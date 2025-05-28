import torch
import torch.nn as nn 
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm

model_dict = {
    'siglip2': 'hf-hub:timm/ViT-B-16-SigLIP2'
}

def extract_features(tag, dataloader, device='mps'):
    """
    Extracts features from images using a specified model and stores the results.

    Parameters:
    - tag (str): The key to select the model from `model_dict`.
    - dataloader (DataLoader): DataLoader providing batches of images.
    - device (str): Device for computation ('cpu' or 'mps'). Default is 'mps'.

    Returns:
    - tuple: A tuple containing:
        * features (np.ndarray): Extracted image features as a numpy array.
        * labels (np.ndarray): Corresponding labels for the features.

    Raises:
    - ValueError: If the tag does not exist in `model_dict`.
    """
    # Dictionary to store models with their tags
    if tag not in model_dict:
        raise ValueError(f"Model for tag '{tag}' not found.")
    
    print(f'Loading Model: {tag}')
    model_name = model_dict[tag]
    model, preprocess = create_model_from_pretrained(model_name)
    model = model.to(device)
    model = model.eval()
    tokenizer = get_tokenizer(model_name)
    print(f'Loaded Model: {tag}')
    print('Beginning Feature Extraction')

    features = []
    labels = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (images, label) in enumerate(tqdm(dataloader)):
            # Forward pass to extract features
            features += [model.forward(images.to(device))[0]]
            labels += [label]

    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    print('feature shape', features.shape)
    print('labels shape', labels.shape)

    return features, labels

@torch.no_grad()
def normalize(input):
    """
    Normalizes the input tensor along its last dimension.

    Parameters:
    - input (Tensor): The input tensor to be normalized.

    Returns:
    - Tensor: The L2-normalized vector.
    """
    input = torch.tensor(input)
    return F.normalize(input, dim=-1)

@torch.no_grad()
def compute_similarities(image_feat, text_feat, logit_scale, logit_bias):
    """
    Computes the similarity between image and text features.

    Parameters:
    - image_feat (Tensor): Image feature tensor.
    - text_feat (Tensor): Text feature tensor.
    - logit_scale (float): Scaling factor for logits.
    - logit_bias (float): Bias term for logits.

    Returns:
    - Tensor: Similarity scores between images and texts.
    """
    return torch.sigmoid(normalize(image_feat) @ normalize(text_feat).T * logit_scale + logit_bias)

def get_word_embeddings(noun_file_path, tokenizer, model, device='mps', batch_size=64):
    """
    Encodes a list of nouns into word embeddings using the specified model.

    Parameters:
    - noun_file_path (str): Path to the CSV file containing nouns.
    - tokenizer: Tokenizer object for processing text inputs.
    - model: Pre-trained model used for encoding texts.
    - device (str): Device on which computations are performed ('cpu' or 'cuda'). Default is 'mps'.
    - batch_size (int): Number of samples processed at once during encoding. Default is 64.

    Returns:
    - Tensor: Concatenated tensor containing all word embeddings.
    """
    # Load nouns from the file
    with open(noun_file_path, 'r') as f:
        nouns = f.read().split('\n')
    
    # Add a prefix to each noun for better context
    tmp_nouns = ['An Image of a ' + noun for noun in nouns]
    
    # Tokenize the preprocessed nouns
    words = tokenizer(tmp_nouns, context_length=model.context_length)
    
    # Encode texts into embeddings in batches
    word_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(words), batch_size)):
            batch_words = words[i:i+batch_size].to(device)
            batch_embeddings = model.encode_text(batch_words).cpu()
            word_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings into a single tensor
    return torch.cat(word_embeddings)

def reset_weights(model):
    """
    Reset weights of the model by reinitializing parameters.
    """
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, latent_dim=10):
        super(ResNet18FeatureExtractor, self).__init__()
        
        # Load a pre-trained ResNet-18 model
        self.base_model = resnet18(pretrained=True)
                
        # Replace the last fully connected layer with a new one for your desired latent dimension
        self.base_model.fc = nn.Identity()  # Temporarily replace to bypass it
        
        # Create a new fully connected layer for your specific task
        self.latent_dim_layer = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        # Forward pass through the base ResNet-18 model up to the penultimate layer
        features = self.base_model(x)
        
        # Pass the extracted features through your custom fully connected layer
        latent_features = self.latent_dim_layer(features)
        
        return latent_features

def build_encoder(latent_dim=10):
    """
    Encoder using ResNet-18 mapping images to latent_dim features.
    """
    # Load a pre-trained ResNet-18 model
    model = ResNet18FeatureExtractor(latent_dim = latent_dim)
    
    return model

def build_decoder(latent_dim=10):
    # The decoder mirrors the ResNet structure but in reverse
    return nn.Sequential(
        nn.Linear(latent_dim, 512 * 2 * 2),  # Adjust to match the size of feature maps before final FC layer in ResNet-18
        nn.ReLU(),
        
        # Upsample and convolve back to original image dimensions
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(256), 
        nn.ReLU(),
        
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # Final layer to produce the image
        nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
        nn.Sigmoid()  # Assuming input images are normalized between 0 and 1
    )

def build_autoencoder(latent_dim=10):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)

    return encoder, decoder